
clear

unset all

echo ""

############################################################

number_heatmap_averaging_windows_list=(10 20 30 40 50 80 100 200 500)

image_filter_sigma_list=(1 2 4 7 10)

local_window_size_list=(1 2 4 7 10 15 20 25 30 35 40 45 50 60 80 100)

############################################################

# number_heatmap_averaging_windows_list=(10 20)

# image_filter_sigma_list=(1 2)

# local_window_size_list=(2 4)

############################################################

for filename in *.tif; do

	echo "File being processed is $filename"

	rm -rf ${filename:0:-4}

	mkdir ${filename:0:-4}

	cp $filename OrientationCoherance2D.py ./${filename:0:-4}

	cd ./${filename:0:-4}

	############################################################

	for i in "${number_heatmap_averaging_windows_list[@]}"; do

	  echo ""
		
	  echo "Current number of heatmap windows are $i"

		for j in "${image_filter_sigma_list[@]}"; do
	    
			for k in "${local_window_size_list[@]}"; do

				sed -e "s/CHANGE_filename/$filename/g" -e "s/CHANGE_heatmap_averaging_windows/$i/g" -e "s/CHANGE_image_filter_sigma/$j/g" -e "s/CHANGE_local_window_size/$k/g" OrientationCoherance2D.py > OrientationCoherance2D_editedH$i,S$j,W$k.py

	            ############################################################

	            python3 OrientationCoherance2D_editedH$i,S$j,W$k.py
	            
	            rm OrientationCoherance2D_editedH$i,S$j,W$k.py

	            ############################################################

			done # endloop for k

		done # endloop for j

	done # endloop for i

	echo ""

	cd ..

	############################################################

done

