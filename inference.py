import torch
import numpy as np
from PIL import Image
import argparse
import os
import time
import json
import csv
import glob
from pathlib import Path

from utils.paths import *
from vision_tower import VGGT_OriAny_Ref
from utils.app_utils import *
from utils.axis_renderer import BlendRenderer


def load_model(device):
    """Load the model with checkpoint"""
    if os.path.exists(LOCAL_CKPT_PATH):
        ckpt_path = LOCAL_CKPT_PATH
    else:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id="Viglong/OriAnyV2_ckpt", 
            filename=HF_CKPT_PATH, 
            repo_type="model", 
            cache_dir='./', 
            resume_download=True
        )
    
    mark_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT_OriAny_Ref(out_dim=900, dtype=mark_dtype, nopretrain=True)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.eval()
    model = model.to(device)
    print('✓ Model loaded.')
    
    return model


def run_inference_script(ref_image_path, tgt_image_path=None, remove_background=True, output_dir='./output'):
    """
    Run inference on images and save results
    
    Args:
        ref_image_path: Path to reference image
        tgt_image_path: Path to target image (optional)
        remove_background: Whether to remove background
        output_dir: Directory to save output files
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    start_total = time.time()
    model = load_model(device)
    
    # Load images
    if not os.path.exists(ref_image_path):
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
    
    pil_ref = Image.open(ref_image_path).convert("RGB")
    print(f'✓ Reference image loaded: {ref_image_path}')
    
    pil_tgt = None
    if tgt_image_path:
        if not os.path.exists(tgt_image_path):
            raise FileNotFoundError(f"Target image not found: {tgt_image_path}")
        pil_tgt = Image.open(tgt_image_path).convert("RGB")
        print(f'✓ Target image loaded: {tgt_image_path}')
    
    # Preprocess (remove background if requested)
    if remove_background:
        print('Processing background removal...')
        pil_ref = background_preprocess(pil_ref, True)
        if pil_tgt:
            pil_tgt = background_preprocess(pil_tgt, True)
    
    # Model inference
    print('Running inference...')
    inference_start = time.time()
    
    with torch.no_grad():
        ans_dict = inf_single_case(model, pil_ref, pil_tgt)
    
    inference_time = time.time() - inference_start
    print(f'✓ Inference completed in {inference_time:.3f}s')
    
    # Extract results
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except:
            return float(default)
    
    az = safe_float(ans_dict.get('ref_az_pred', 0))
    el = safe_float(ans_dict.get('ref_el_pred', 0))
    ro = safe_float(ans_dict.get('ref_ro_pred', 0))
    alpha = int(ans_dict.get('ref_alpha_pred', 1))
    
    # Print results
    print('\n' + '='*60)
    print('RESULTS - Reference Image')
    print('='*60)
    print(f'Azimuth (0~360°):    {az:.2f}°')
    print(f'Polar (-90~90°):     {el:.2f}°')
    print(f'Rotation (-90~90°):  {ro:.2f}°')
    print(f'Num Directions:      {alpha}')
    
    results = {
        'reference': {
            'azimuth': az,
            'polar': el,
            'rotation': ro,
            'num_directions': alpha
        }
    }
    
    # Handle target image if provided
    if pil_tgt is not None:
        rel_az = safe_float(ans_dict.get('rel_az_pred', 0))
        rel_el = safe_float(ans_dict.get('rel_el_pred', 0))
        rel_ro = safe_float(ans_dict.get('rel_ro_pred', 0))
        
        tgt_azi, tgt_ele, tgt_rot = Get_target_azi_ele_rot(az, el, ro, rel_az, rel_el, rel_ro)
        
        print('\n' + '='*60)
        print('RESULTS - Target Image')
        print('='*60)
        print(f'Azimuth (0~360°):    {tgt_azi:.2f}°')
        print(f'Polar (-90~90°):     {tgt_ele:.2f}°')
        print(f'Rotation (-90~90°):  {tgt_rot:.2f}°')
        
        print('\n' + '='*60)
        print('RESULTS - Relative Pose')
        print('='*60)
        print(f'Relative Azimuth:    {rel_az:.2f}°')
        print(f'Relative Polar:      {rel_el:.2f}°')
        print(f'Relative Rotation:   {rel_ro:.2f}°')
        
        results['target'] = {
            'azimuth': tgt_azi,
            'polar': tgt_ele,
            'rotation': tgt_rot
        }
        results['relative'] = {
            'azimuth': rel_az,
            'polar': rel_el,
            'rotation': rel_ro
        }
    
    # Render and overlay axis
    print('\nRendering axis visualization...')
    axis_renderer = BlendRenderer(RENDER_FILE)
    
    import tempfile
    tmp_ref = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_ref.close()
    
    render_start = time.time()
    
    # Render reference
    axis_renderer.render_axis(az, el, ro, alpha, save_path=tmp_ref.name)
    axis_ref = Image.open(tmp_ref.name).convert("RGBA")
    
    if axis_ref.size != pil_ref.size:
        pil_ref = pil_ref.resize(axis_ref.size, Image.BICUBIC)
    pil_ref_rgba = pil_ref.convert("RGBA")
    overlaid_ref = Image.alpha_composite(pil_ref_rgba, axis_ref).convert("RGB")
    
    # Save reference output
    ref_output_path = os.path.join(output_dir, 'reference_overlay.png')
    overlaid_ref.save(ref_output_path)
    print(f'✓ Reference overlay saved: {ref_output_path}')
    
    # Handle target overlay if provided
    if pil_tgt is not None:
        tmp_tgt = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_tgt.close()
        
        axis_renderer.render_axis(tgt_azi, tgt_ele, tgt_rot, alpha=1, save_path=tmp_tgt.name)
        axis_tgt = Image.open(tmp_tgt.name).convert("RGBA")
        
        if axis_tgt.size != pil_tgt.size:
            pil_tgt = pil_tgt.resize(axis_tgt.size, Image.BICUBIC)
        pil_tgt_rgba = pil_tgt.convert("RGBA")
        overlaid_tgt = Image.alpha_composite(pil_tgt_rgba, axis_tgt).convert("RGB")
        
        tgt_output_path = os.path.join(output_dir, 'target_overlay.png')
        overlaid_tgt.save(tgt_output_path)
        print(f'✓ Target overlay saved: {tgt_output_path}')
        
        os.remove(tmp_tgt.name)
    
    render_time = time.time() - render_start
    os.remove(tmp_ref.name)
    
    # Save JSON results
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'✓ Results saved: {json_path}')
    
    # Timing summary
    total_time = time.time() - start_total
    print('\n' + '='*60)
    print('TIMING SUMMARY')
    print('='*60)
    print(f'Model Inference:     {inference_time:.3f}s')
    print(f'Rendering + Overlay: {render_time:.3f}s')
    print(f'Total Time:          {total_time:.3f}s')
    print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Orient-Anything V2 Inference Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python inference.py --ref assets/examples/F35-0.jpg
  
  # Reference and target image with background removal
  python inference.py --ref assets/examples/F35-0.jpg --tgt assets/examples/F35-1.jpg
  
  # Batch processing - all images in directory
  python inference.py --ref-dir dataset/
  
  # Batch processing with custom pattern (e.g., only .jpg files)
  python inference.py --ref-dir dataset/ --pattern "*.jpg"
  
  # Without background removal
  python inference.py --ref assets/examples/bottle.jpg --no-remove-bg
  
  # Custom output directory
  python inference.py --ref assets/examples/F35-0.jpg --output results/
        """
    )
    
    parser.add_argument('--ref', type=str, default=None, help='Path to reference image (single image mode)')
    parser.add_argument('--ref-dir', type=str, default=None, help='Path to directory with images (batch mode)')
    parser.add_argument('--tgt', type=str, default=None, help='Path to target image (optional, single mode only)')
    parser.add_argument('--pattern', type=str, default='*', help='File pattern for batch mode (default: * for all files)')
    parser.add_argument('--no-remove-bg', action='store_true', help='Disable background removal')
    parser.add_argument('--output', type=str, default='./output', help='Output directory (default: ./output)')
    
    args = parser.parse_args()
    
    try:
        # Batch processing mode
        if args.ref_dir:
            if not os.path.isdir(args.ref_dir):
                raise ValueError(f"Directory not found: {args.ref_dir}")
            
            print(f"🔄 Batch Processing Mode - Directory: {args.ref_dir}")
            print(f"📁 Pattern: {args.pattern}\n")
            
            # Find all images matching pattern
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
            image_files = []
            
            for ext in image_extensions:
                pattern = os.path.join(args.ref_dir, args.pattern if args.pattern != '*' else ext)
                image_files.extend(glob.glob(pattern))
            
            # Remove duplicates and sort
            image_files = sorted(list(set(image_files)))
            
            if not image_files:
                raise ValueError(f"No images found in {args.ref_dir} with pattern {args.pattern}")
            
            print(f"✓ Found {len(image_files)} image(s)\n")
            
            # Setup device and load model once
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Using device: {device}\n')
            
            model = load_model(device)
            axis_renderer = BlendRenderer(RENDER_FILE)
            
            # Create output structure
            os.makedirs(args.output, exist_ok=True)
            overlays_dir = os.path.join(args.output, 'overlays')
            os.makedirs(overlays_dir, exist_ok=True)
            
            # Results storage
            all_results = []
            csv_path = os.path.join(args.output, 'batch_results.csv')
            json_path = os.path.join(args.output, 'batch_results.json')
            
            batch_start = time.time()
            
            # Process each image
            for idx, image_path in enumerate(image_files, 1):
                print(f"\n{'='*60}")
                print(f"Processing [{idx}/{len(image_files)}]: {os.path.basename(image_path)}")
                print(f"{'='*60}")
                
                try:
                    # Load image
                    pil_ref = Image.open(image_path).convert("RGB")
                    print(f'✓ Image loaded')
                    
                    # Preprocess
                    if not args.no_remove_bg:
                        pil_ref = background_preprocess(pil_ref, True)
                    
                    # Inference
                    inference_start = time.time()
                    with torch.no_grad():
                        ans_dict = inf_single_case(model, pil_ref, None)
                    inference_time = time.time() - inference_start
                    
                    # Extract results
                    def safe_float(val, default=0.0):
                        try:
                            return float(val)
                        except:
                            return float(default)
                    
                    az = safe_float(ans_dict.get('ref_az_pred', 0))
                    el = safe_float(ans_dict.get('ref_el_pred', 0))
                    ro = safe_float(ans_dict.get('ref_ro_pred', 0))
                    alpha = int(ans_dict.get('ref_alpha_pred', 1))
                    
                    print(f'Azimuth:  {az:.2f}° | Polar: {el:.2f}° | Rotation: {ro:.2f}° | Directions: {alpha}')
                    print(f'Inference time: {inference_time:.3f}s')
                    
                    # Render overlay
                    render_start = time.time()
                    import tempfile
                    tmp_ref = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    tmp_ref.close()
                    
                    axis_renderer.render_axis(az, el, ro, alpha, save_path=tmp_ref.name)
                    axis_ref = Image.open(tmp_ref.name).convert("RGBA")
                    
                    if axis_ref.size != pil_ref.size:
                        pil_ref = pil_ref.resize(axis_ref.size, Image.BICUBIC)
                    pil_ref_rgba = pil_ref.convert("RGBA")
                    overlaid = Image.alpha_composite(pil_ref_rgba, axis_ref).convert("RGB")
                    
                    # Save overlay
                    overlay_filename = f"{idx:03d}_{Path(image_path).stem}_overlay.png"
                    overlay_path = os.path.join(overlays_dir, overlay_filename)
                    overlaid.save(overlay_path)
                    
                    render_time = time.time() - render_start
                    os.remove(tmp_ref.name)
                    
                    print(f'✓ Overlay saved: {overlay_filename}')
                    
                    # Store results
                    result = {
                        'filename': os.path.basename(image_path),
                        'azimuth': az,
                        'polar': el,
                        'rotation': ro,
                        'num_directions': alpha,
                        'inference_time': inference_time,
                        'render_time': render_time,
                        'total_time': inference_time + render_time
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    print(f'✗ Error processing {os.path.basename(image_path)}: {e}')
                    result = {
                        'filename': os.path.basename(image_path),
                        'error': str(e)
                    }
                    all_results.append(result)
            
            total_batch_time = time.time() - batch_start
            
            # Save CSV results
            if all_results and 'error' not in all_results[0]:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['filename', 'azimuth', 'polar', 'rotation', 'num_directions', 'inference_time', 'render_time', 'total_time'])
                    writer.writeheader()
                    for result in all_results:
                        if 'error' not in result:
                            writer.writerow(result)
                print(f'\n✓ CSV results saved: {csv_path}')
            
            # Save JSON results
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f'✓ JSON results saved: {json_path}')
            
            # Batch summary
            successful = sum(1 for r in all_results if 'error' not in r)
            failed = len(all_results) - successful
            
            print(f'\n{"="*60}')
            print('BATCH PROCESSING SUMMARY')
            print(f'{"="*60}')
            print(f'Total images: {len(image_files)}')
            print(f'Successful: {successful}')
            print(f'Failed: {failed}')
            print(f'Total batch time: {total_batch_time:.3f}s')
            print(f'Average time per image: {total_batch_time/len(image_files):.3f}s')
            print(f'{"="*60}')
            
            if successful > 0:
                avg_inference = np.mean([r['inference_time'] for r in all_results if 'error' not in r])
                avg_render = np.mean([r['render_time'] for r in all_results if 'error' not in r])
                print(f'Average inference time: {avg_inference:.3f}s')
                print(f'Average render time: {avg_render:.3f}s')
                print(f'{"="*60}')
            
            print(f'\n✓ Batch processing completed!')
            print(f'📁 Results saved to: {args.output}')
        
        # Single image mode
        elif args.ref:
            run_inference_script(
                ref_image_path=args.ref,
                tgt_image_path=args.tgt,
                remove_background=not args.no_remove_bg,
                output_dir=args.output
            )
            print('\n✓ Inference completed successfully!')
        
        else:
            parser.print_help()
            print("\n✗ Error: Please provide either --ref (single image) or --ref-dir (batch mode)")
            exit(1)
            
    except Exception as e:
        print(f'\n✗ Error: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
