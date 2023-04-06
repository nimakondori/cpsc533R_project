#!/bin/bash
# CUDA version from the command-line argument
command=${1:-build}
cuda_version=${2:-11.6}
torch_version=${3:-1.13.1}
final_tag=nimakondori/cpsc_533r_proj:torch_${torch_version}_cuda_${cuda_version}_final

echo "Final Tag == $final_tag"

if [[ "$command" == "build" ]]; then
	docker build --build-arg cuda_version=$cuda_version --build-arg torch_version=$torch_version --tag $final_tag .

	docker push $final_tag
elif [[ "$command" == "run" ]]; then  
	
	 docker run -it -d \
	          --gpus device=ALL \
     	      --name=victoria_cpsc533r_project3  \
      	      --volume=$HOME/cpsc533R_project:/workspace/cpsc533R_project \
      	      --volume=/mnt/nas-server/workspace/mobina/LV_PLAX2_cleaned/Cleaned:/mnt/data/LV_PLAX2_cleaned/Cleaned \
      	      --volume=/mnt/nas-server/workspace/mohammadj/Datasets/Mohammadj/LVOT_LVD_Landmark/PLAX_LVOT_landmarks:/mnt/data/PLAX_LVOT_landmarks \
      	      $final_tag
else
	echo "invalid command. Use build or run"
fi 
