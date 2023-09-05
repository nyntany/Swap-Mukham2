# Swap-Mukham WIP

## Changes
- added parellel execution utilizing concurrent.futures
- added test preview button
- added video frame slider
- added collect face from preview
- added frame selection to replace trim feature
- converted every model to onnx format
- editable onnx execution providers at ``utils/device``
- removed moviepy, insightface dependency
- removed pytorch dependency for cpu users
- command line arg ``--prefer_text_widget`` to replace target video widget with text
- more face-upscaler support (restoreformer, codeformer, gfpgan, gpen)
- added face-upscaler opacity slider
- nsfw-detector now only checks random 100 frames
- added custom foreground painting
- ui changes (gradio 3.40)
- added swap-iteration (may increase face likeliness)
- added date-time option for output name

  
## Install WIP branch
### CPU Install
````
git clone -b new-wip --single-branch https://github.com/harisreedhar/Swap-Mukham.git
cd Swap-Mukham
conda create -n swapmukham python=3.10 -y
conda activate swapmukham
pip install -r requirements_cpu.txt

python app.py --cpu --prefer_text_widget
````
### GPU Install (CUDA)
````
git clone -b new-wip --single-branch https://github.com/harisreedhar/Swap-Mukham.git
cd Swap-Mukham
conda create -n swapmukham python=3.10 -y
conda activate swapmukham
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

python app.py --prefer_text_widget
````

## Download and place these models under ``assets/pretrained_models``

- [inswapper_128.onnx](https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx)
- [faceparser.onnx](https://huggingface.co/bluefoxcreation/Face_parsing_onnx/resolve/main/faceparser.onnx)
- [det_10g.onnx](https://huggingface.co/bluefoxcreation/insightface-retinaface-arcface-model/resolve/main/det_10g.onnx)
- [w600k_r50.onnx](https://huggingface.co/bluefoxcreation/insightface-retinaface-arcface-model/resolve/main/w600k_r50.onnx)

#### Face upscalers ``optional``
- [codeformer.onnx](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/codeformer.onnx)
- [GFPGANv1.2.onnx](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GFPGANv1.2.onnx)
- [GFPGANv1.3.onnx](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GFPGANv1.3.onnx)
- [GFPGANv1.4.onnx](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GFPGANv1.4.onnx)
- [GPEN-BFR-256.onnx](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-256.onnx)
- [GPEN-BFR-512.onnx](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/GPEN-BFR-512.onnx)
- [restoreformer.onnx](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/Models/restoreformer.onnx)
