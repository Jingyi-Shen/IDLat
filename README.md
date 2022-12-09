# QualityAE

Code for "IDLat: An Importance-Driven Latent Generation Method for Scientific Data", IEEE Transactions on Visualization and Computer Graphics (IEEE VIS 2022). 

<!--
![Python 3.7](https://img.shields.io/badge/Python-3.7-green.svg?style=plastic)
![pytorch 1.1.0](https://img.shields.io/badge/Pytorch-1.1.0-green.svg?style=plastic)
![pyqt5 5.13.0](https://img.shields.io/badge/pyqt5-5.13.0-green.svg?style=plastic)
-->

To train IDLat on Vortex dataset, run 

```

mkdir ./results/vortex_600_beta5_a7
OUTDIR=./results/vortex_600_beta5_a7
python -u train.py --config=./configs/config_vortex.yaml --name=vortex_600_beta5_a7 --train  > ${OUTDIR}/vortex_600_beta5_a7.log

```

To evalute IDLat on Vortex dataset and generate latent representations with uniform Importance Map (e.g., Importance value = 0.9), run 

```

python eval.py --config=./configs/config_vortex.yaml \
               --snapshot ./results/vortex_600_beta5_a7/snapshots/best.pt \
               --tqdm \
               --output_dir ./results/vortex_600_beta5_a7/outputs/ \
               --map_value 0.9 \
               --map_name 'uni09' 

```

To evalute IDLat on Vortex dataset and generate latent representations with isosurface distance map (e.g., isovalue = 7), run 

```
python eval.py --config=./configs/vortex/config_vortex2.yaml \
               --snapshot ./results/vortex_600_beta5_a7/snapshots/best.pt \
               --tqdm \
               --output_dir ./results/vortex_600_beta5_a7/outputs/ \
               --map_name 'iso7' 

```

Please modify configure file as needed.



## Citation

If you use this code for your research, please cite our paper.
```
@ARTICLE{shen2022IDLat,  
  title={IDLat: An Importance-Driven Latent Generation Method for Scientific Data},   
  author={Shen, Jingyi and Li, Haoyu and Xu, Jiayi and Biswas, Ayan and Shen, Han-Wei},  
  journal={IEEE Transactions on Visualization and Computer Graphics},   
  year={2022},  
  pages={1-11},  
  doi={10.1109/TVCG.2022.3209419}}
```

