
clear

unset all


############################################################

image_filter_sigma_list=(1 2 5 10)

local_window_size_list=(1 2 5 7 10 15 20 25 30 40 50)

# image_filter_sigma_list=(1 2)

# local_window_size_list=(2 5)

############################################################

for filename in *.tif; do

	echo "File being processed is $filename"

	echo ""

	rm -rf ${filename:0:-4}

	mkdir ${filename:0:-4}

	cp $filename OrientationCoherance2D.py ./${filename:0:-4}

	cd ./${filename:0:-4}

	############################################################

	for j in "${image_filter_sigma_list[@]}"; do

    
		for k in "${local_window_size_list[@]}"; do

			sed -e "s/CHANGE_filename/$filename/g" -e "s/CHANGE_image_filter_sigma/$j/g" -e "s/CHANGE_local_window_size/$k/g" OrientationCoherance2D.py > OrientationCoherance2D_editedS$j,W$k.py

            ############################################################

            python3 OrientationCoherance2D_editedS$j,W$k.py
            
            rm OrientationCoherance2D_editedS$j,W$k.py

            ############################################################

            echo "Sigma processed is $j, window size processed is $k"

            ############################################################

		done # endloop for k

	done # endloop for j

	echo ""

	echo "================================"

	echo ""

	cd ..

	############################################################

done
