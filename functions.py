import os
import sys
import load as loader
import upscale as upscaler
import pbr as materializer
import write as writer
import config
from tqdm import tqdm
import pathlib
from zipfile import ZipFile
import shutil

import ai.ESRGAN.upscaler
import ai.RealESRGAN.upscaler

from generation.normals import *
from generation.metalness import *
from generation.roughness import *

from ai.PBR.model import Unet
import ai.PBR.eval_disp as displacements
import ai.PBR.eval_norm as normals
import ai.PBR.eval_rough as roughness
sys.path
sys.path.append('./nvidia')
from octahedral import *

def normal_single( texture ):
   textureUpscaled = os.path.exists(f"textures/processing/upscaled/{texture}.png")
   path = f"textures/processing/diffuse/{texture}.png"
   if( textureUpscaled ):
      path = f"textures/processing/upscaled/{texture}.png"

   generate_normal(
      path,
      f"textures/processing/normals/{texture}_normal.png",0,1.5)
   return "Done!"

def roughness_single( texture ):
   textureUpscaled = os.path.exists(f"textures/processing/upscaled/{texture}.png")
   path = f"textures/processing/diffuse/{texture}.png"
   if( textureUpscaled ):
      path = f"textures/processing/upscaled/{texture}.png"

   generate_roughness(
      path,
      f"textures/processing/roughness/{texture}_rough.png",1)
   return "Done!"

def metalness_single( texture ):
   textureUpscaled = os.path.exists(f"textures/processing/upscaled/{texture}.png")
   path = f"textures/processing/diffuse/{texture}.png"
   if( textureUpscaled ):
      path = f"textures/processing/upscaled/{texture}.png"

   generate_metalness(
      path,
      f"textures/processing/metallness/{texture}_metal.png",1)
   return "Done!"

def textures_list():
   Upscaled = os.listdir("textures/processing/upscaled/")
   Diffuse = os.listdir("textures/processing/diffuse/")
   array_files = []
   
   for x in range(len(Upscaled)):
      temp = Upscaled[x]
      array_files.append(temp)

   for x in range(len(Diffuse)):
      temp = Diffuse[x]
      if( temp in array_files ):
         continue
      array_files.append(temp)

   return ','.join(array_files)

def tabs_list():
   tabFiles = os.listdir("webui/tabs/")
   array_files = []
   
   for x in range(len(tabFiles)):
      temp = tabFiles[x]
      array_files.append(temp)

   array_files = sorted(array_files, key=lambda x: x[0]) 

   return ','.join(array_files)

def unupscale( texture ):
   os.remove(f"textures/processing/upscaled/{texture}.png")
   return "Removed!"

def remove_all_pbr( texture ):
   try:
      os.remove(f"textures/processing/normals/{texture}_normal.png")
   except Exception as e:
      print(e)

   try:
      os.remove(f"textures/processing/roughness/{texture}_rough.png")
   except Exception as e:
      print(e)

   try:
      os.remove(f"textures/processing/metallness/{texture}_metal.png")
   except Exception as e:
      print(e)

   try:
      os.remove(f"textures/processing/displacements/{texture}_disp.png")
   except Exception as e:
      print(e)

   return "Removed!"

def ai_normal_single( texture ):
   textureUpscaled = os.path.exists(f"textures/processing/upscaled/{texture}.png")
   path = f"textures/processing/diffuse/{texture}.png"
   if( textureUpscaled ):
      path = f"textures/processing/upscaled/{texture}.png"

   import gc
   import torch
   torch.cuda.empty_cache()
   gc.collect()

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   PATH_CHK = "ai/PBR/checkpoints/norm/norm_net_last.pth"

   norm_net = Unet().to(device)
   checkpoint = torch.load(PATH_CHK)
   norm_net.load_state_dict(checkpoint["model"])

   normals.generateNormSingle(norm_net,path,"textures/processing/normaldx")
   for x in tqdm( os.listdir(f"textures/processing/normaldx/"), desc="Generating..." ):
      if x.endswith(".png"):
         LightspeedOctahedralConverter.convert_dx_file_to_octahedral(f"textures/processing/normaldx/{x}", f"textures/processing/normals/{x}")

   return "Normal map is done!"

def ai_roughness_single( texture ):
   textureUpscaled = os.path.exists(f"textures/processing/upscaled/{texture}.png")
   path = f"textures/processing/diffuse/{texture}.png"
   if( textureUpscaled ):
      path = f"textures/processing/upscaled/{texture}.png"

   import gc
   import torch
   torch.cuda.empty_cache()
   gc.collect()

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   PATH_CHK = "ai/PBR/checkpoints/rough/rough_net_last.pth"

   norm_net = Unet().to(device)
   checkpoint = torch.load(PATH_CHK)
   norm_net.load_state_dict(checkpoint["model"])

   roughness.generateRoughSingle(norm_net,path,"textures/processing/roughness")
   return "Roughness map is done!"

def ai_parallax_single( texture ):
   textureUpscaled = os.path.exists(f"textures/processing/upscaled/{texture}.png")
   path = f"textures/processing/diffuse/{texture}.png"
   if( textureUpscaled ):
      path = f"textures/processing/upscaled/{texture}.png"

   import gc
   import torch
   torch.cuda.empty_cache()
   gc.collect()

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   PATH_CHK = "ai/PBR/checkpoints/disp/disp_net_last.pth"

   norm_net = Unet().to(device)
   checkpoint = torch.load(PATH_CHK)
   norm_net.load_state_dict(checkpoint["model"])

   displacements.generateDispSingle(norm_net,path,"textures/processing/displacements")
   return "Displacement map is done!"

def plugins_list():
   tabFiles = os.listdir("plugins/")
   array_files = []
   
   for x in range(len(tabFiles)):
      temp = tabFiles[x]
      array_files.append(temp)

   array_files = sorted(array_files, key=lambda x: x[0]) 

   return ','.join(array_files)

def test( texture ):
   return texture

def upscale_single4( texture ):
   try:
      if( "Real" in config.upscale_model ):
         ai.RealESRGAN.upscaler.upscaleFile( texture )
      else:
         ai.ESRGAN.upscaler.upscaleFile( texture )
      return "Upscaled!"
   except Exception as e:
      return str(e)

def upscale_single8( texture ):
   try:

      print("Upscaling First time")

      if( "Real" in config.upscale_model ):
         ai.RealESRGAN.upscaler.upscaleFile( texture )
      else:
         ai.ESRGAN.upscaler.upscaleFile( texture )

      isExist = os.path.exists("textures/processing/lowres")
      if not isExist:
         os.makedirs("textures/processing/lowres")

      shutil.move(f"textures/processing/diffuse/{texture}.png", f"textures/processing/lowres/{texture}.png")   
      shutil.move(f"textures/processing/upscaled/{texture}.png", f"textures/processing/diffuse/{texture}.png")
            

      print("Upscaling second time")
      if( "Real" in config.upscale_model ):
         ai.RealESRGAN.upscaler.upscaleFile( texture )
      else:
         ai.ESRGAN.upscaler.upscaleFile( texture )

      shutil.move(f"textures/processing/lowres/{texture}.png", f"textures/processing/diffuse/{texture}.png")


      return "Upscaled!"
   except Exception as e:
      return str(e)

def load_captures():
   loadDirId = 0
   loadDir = "captures"

   mods = [f for f in pathlib.Path().glob(f"{config.rtx_remix_dir}/mods/*")]
   if( len(mods) > 0 ):
      print("Directories for importing: ")

      print("0 | captures")
      for x in range(len(mods)):
         mod_path = str(mods[x]).replace(f"{config.rtx_remix_dir}/mods/","")
         print(f"{x+1} | {mod_path}")


      #loadDirId = int(input("From what dir you want to load the textures, select by index: "))
      loadDirId = 0
      loadDirId -= 1
      if( loadDirId != -1 ):
         loadDir = str(mods[loadDirId]).replace(f"{config.rtx_remix_dir}/","") + "/SubUSDs/textures/diffuse"
      else:
         loadDir += "/textures"
   else:
      loadDir += "/textures"


   loaded = loader.loadTextures(loadDir,0,0)
   if( loaded < 1 ):
      return f"Oops, can't load the textures!"

   return f"{loaded} textures loaded!"

def write_mod():
   mod_dir = "OctoTexGUI"
   isExist = os.path.exists(f"{config.rtx_remix_dir}/mods/{mod_dir}")
   if not isExist:
      with ZipFile("mods/modTemplate.zip", 'r') as zObject:
         zObject.extractall( path=f"{config.rtx_remix_dir}/mods/")

      os.rename(f"{config.rtx_remix_dir}/mods/modTemplate", f"{config.rtx_remix_dir}/mods/{mod_dir}")

   replacements_file = "replacements.usda"
   written = writer.saveAllTextures(mod_dir, replacements_file)
   return f"{written} textures has been written!"

def clear_hashes():
   f = open("hash/hashes.json", "w")
   f.write("""
{
   "loaded": {},
   "upscaled": {},
   "materialized": {},
   "saved": {}
}""")

   f.close()
   return "success!"


def upscale_all_4():
   upscaler.upscaleTextures()
   return "Global upscaling is done!"

def upscale_all_8():
   upscaler.upscaleTextures2X()
   return "Global upscaling is done!"

def generate_pbr():
   materializer.generatePBR()
   return "Global PBR generation is done!"

def generate_pbr_ai():
   import gc
   import torch
   torch.cuda.empty_cache()
   gc.collect()

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   PATH_CHK = "ai/PBR/checkpoints/norm/norm_net_last.pth"

   norm_net = Unet().to(device)
   checkpoint = torch.load(PATH_CHK)
   norm_net.load_state_dict(checkpoint["model"])

   normals.generateNorm(norm_net,"textures/processing/upscaled","textures/processing/normaldx")
   for x in tqdm( os.listdir(f"textures/processing/normaldx/"), desc="Generating..." ):
      if x.endswith(".png"):
         LightspeedOctahedralConverter.convert_dx_file_to_octahedral(f"textures/processing/normaldx/{x}", f"textures/processing/normals/{x}")


   torch.cuda.empty_cache()
   gc.collect()

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   PATH_CHK = "ai/PBR/checkpoints/rough/rough_net_last.pth"

   norm_net = Unet().to(device)
   checkpoint = torch.load(PATH_CHK)
   norm_net.load_state_dict(checkpoint["model"])

   roughness.generateRough(norm_net,"textures/processing/upscaled","textures/processing/roughness")
   torch.cuda.empty_cache()
   gc.collect()

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   PATH_CHK = "ai/PBR/checkpoints/disp/disp_net_last.pth"

   norm_net = Unet().to(device)
   checkpoint = torch.load(PATH_CHK)
   norm_net.load_state_dict(checkpoint["model"])

   displacements.generateDisp(norm_net,"textures/processing/upscaled","textures/processing/displacements")

   return "Global AI PBR generation is done!"