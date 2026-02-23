import gradio as gr
import numpy as np
import torch
import cv2
import os
import sys
import rembg
import trimesh
import asyncio
from PIL import Image
from datetime import datetime

# --- AUTOMESH PATH SETUP ---
# Automatically finds the 'libs' folder relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIBS_DIR = os.path.join(BASE_DIR, "libs")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add libraries to Python path
sys.path.insert(0, os.path.join(LIBS_DIR, "ZoeDepth"))
sys.path.insert(0, os.path.join(LIBS_DIR, "TripoSR"))

# --- IMPORTS ---
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

# --- GLOBAL MODELS ---
models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    if "zoe" in models and "sam_auto" in models and "tripo" in models: return
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading AutoMesh Models on {device}...")

    # 1. ZoeDepth (Geometry)
    if "zoe" not in models:
        conf = get_config("zoedepth", "infer")
        conf.pretrained_resource = None
        model_zoe = build_model(conf).to(device).eval()
        
        # Look in weights folder first, then cache
        zoe_ckpt = os.path.join(WEIGHTS_DIR, "ZoeD_M12_N.pt")
        if not os.path.exists(zoe_ckpt):
            # Fallback to cache if user didn't move it
            zoe_ckpt = r"C:\Users\krant\.cache\torch\hub\checkpoints\ZoeD_M12_N.pt"
            
        if os.path.exists(zoe_ckpt):
            print(f"   -> Loading ZoeDepth from {zoe_ckpt}")
            state = torch.load(zoe_ckpt, map_location=device)
            if 'model' in state: model_zoe.load_state_dict(state['model'], strict=False)
            else: model_zoe.load_state_dict(state, strict=False)
        models["zoe"] = model_zoe

    # 2. SAM 2 (Segmentation)
    if "sam_auto" not in models:
        sam_ckpt = os.path.join(WEIGHTS_DIR, "sam2.1_hiera_large.pt")
        # Fallback to old location if move failed
        if not os.path.exists(sam_ckpt):
             sam_ckpt = os.path.join(LIBS_DIR, "segment-anything-2", "checkpoints", "sam2.1_hiera_large.pt")
             
        print(f"   -> Loading SAM 2 from {sam_ckpt}")
        sam_cfg = "sam2.1_hiera_l.yaml"
        sam_model = build_sam2(sam_cfg, sam_ckpt, device=device)
        models["sam_auto"] = SAM2AutomaticMaskGenerator(
            sam_model, points_per_side=32, pred_iou_thresh=0.7, stability_score_thresh=0.85, crop_n_layers=0
        )

    # 3. TripoSR (3D Generation)
    if "tripo" not in models:
        print("   -> Loading TripoSR...")
        models["tripo"] = TSR.from_pretrained(
            "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
        ).to(device)

def get_depth_map(image_pil):
    return models["zoe"].infer_pil(image_pil)

# --- CORE LOGIC: SOLID ROOM ---
def generate_solid_room(depth_meters, image_np):
    """Generates a walkable triangle mesh from depth map"""
    h, w = depth_meters.shape
    scale = 4 
    z = depth_meters[::scale, ::scale]
    img_small = image_np[::scale, ::scale]
    h_s, w_s = z.shape
    
    # Grid
    x = np.arange(w_s)
    y = np.arange(h_s)
    xv, yv = np.meshgrid(x, y)
    
    # Intrinsics
    fx, fy = 1000.0 / scale, 1000.0 / scale
    cx, cy = w_s / 2, h_s / 2
    
    # Reproject
    x_3d = (xv - cx) * z / fx
    y_3d = (yv - cy) * z / fy
    vertices = np.stack([x_3d, y_3d, z], axis=-1).reshape(-1, 3)
    colors = img_small.reshape(-1, 3) / 255.0
    
    # Triangulate
    idx = np.arange(h_s * w_s).reshape(h_s, w_s)
    v1 = idx[:-1, :-1].flatten()
    v2 = idx[1:, :-1].flatten()
    v3 = idx[:-1, 1:].flatten()
    f1 = np.stack([v1, v3, v2], axis=1)
    
    v4 = idx[1:, 1:].flatten()
    f2 = np.stack([v2, v3, v4], axis=1)
    faces = np.vstack([f1, f2])
    
    # Cleanup Long Edges (Shadows)
    v_f = vertices[faces]
    edge_limit = 0.3 # meters
    mask = (np.linalg.norm(v_f[:,0]-v_f[:,1], axis=1) < edge_limit) & \
           (np.linalg.norm(v_f[:,1]-v_f[:,2], axis=1) < edge_limit) & \
           (np.linalg.norm(v_f[:,2]-v_f[:,0], axis=1) < edge_limit)
           
    return trimesh.Trimesh(vertices=vertices, faces=faces[mask], vertex_colors=colors)

def generate_object(crop_img, depth, center_uv, intrinsics):
    # Remove BG
    rembg_session = rembg.new_session()
    no_bg = rembg.remove(crop_img, session=rembg_session)
    fg = resize_foreground(no_bg, 0.85)
    
    # RGB
    white_bg = Image.new("RGB", fg.size, (255, 255, 255))
    white_bg.paste(fg, mask=fg.split()[3])
    
    # Gen Mesh
    with torch.no_grad():
        codes = models["tripo"]([white_bg], device=device)
        mesh = models["tripo"].extract_mesh(codes, has_vertex_color=True)[0]
    
    # Place
    u, v = center_uv
    fx, fy, cx, cy = intrinsics
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    transform[0,0] = transform[1,1] = transform[2,2] = 1.3
    mesh.apply_transform(transform)
    return mesh

def run_pipeline(input_image):
    if input_image is None: return "Please upload an image.", None
    load_models()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    img_pil = Image.fromarray(input_image).convert("RGB")
    
    # 1. Depth & Room
    print("1. Generating Walkable Room...")
    depth_res = get_depth_map(img_pil)
    depth_npy = depth_res.squeeze().cpu().numpy() if isinstance(depth_res, torch.Tensor) else depth_res
    
    room_mesh = generate_solid_room(depth_npy, input_image)
    full_scene = trimesh.Scene()
    full_scene.add_geometry(room_mesh)
    
    # 2. Objects
    print("2. Populating Furniture...")
    masks = models["sam_auto"].generate(input_image)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    intrinsics = (1000.0, 1000.0, input_image.shape[1]/2, input_image.shape[0]/2)
    
    count = 0
    for i, mask_data in enumerate(masks[:10]): # Limit 10 objects
        area = mask_data['area']
        total = input_image.shape[0] * input_image.shape[1]
        
        # Filter Walls/Noise
        if area > total * 0.4 or area < total * 0.02: continue 
        
        y_idx, x_idx = np.where(mask_data['segmentation'])
        y1, y2, x1, x2 = y_idx.min(), y_idx.max(), x_idx.min(), x_idx.max()
        
        # Padding
        pad = 30
        y1 = max(0, y1-pad); y2 = min(img_pil.height, y2+pad)
        x1 = max(0, x1-pad); x2 = min(img_pil.width, x2+pad)
        
        try:
            print(f"   -> Object {i} (Area: {area})")
            center = ((x1+x2)/2, (y1+y2)/2)
            crop = img_pil.crop((x1, y1, x2, y2))
            avg_depth = np.median(depth_npy[y_idx, x_idx])
            
            mesh = generate_object(crop, avg_depth, center, intrinsics)
            full_scene.add_geometry(mesh, node_name=f"obj_{i}")
            count += 1
        except Exception as e:
            print(f"      Skipped: {e}")

    # 3. Export
    # Fix Orientation for Unity (Flip 180 on X)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    full_scene.apply_transform(rot)
    
    final_path = os.path.join(run_dir, "AutoMesh_Scene.glb")
    full_scene.export(final_path)
    return f"Done! Walkable Room + {count} Objects.\nSaved to: {final_path}", final_path

# --- UI ---
iface = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Image(type="numpy", label="Upload Room Photo"),
    outputs=[gr.Textbox(label="Log"), gr.Model3D(label="3D Viewer", clear_color=[0.8, 0.8, 0.8, 1.0])],
    title="AutoMesh v1.0"
)

if __name__ == "__main__":
    try: asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except: pass
    iface.launch(server_name="127.0.0.1", server_port=7860)