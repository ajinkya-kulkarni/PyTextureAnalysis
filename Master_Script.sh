
clear

unset all

echo ""

############################################################

local_window_size_list=(1 2 5 10 20 30 40 50 60 70 80 90 100 120 150 200 250 300 350 400 450 500)

# local_window_size_list=(2 5)

############################################################

for filename in *.tif; do

	echo "File being processed is $filename"

	rm -rf ${filename:0:-4}

	mkdir ${filename:0:-4}

	cp $filename OrientationCoherance2D.py ./${filename:0:-4}

	cd ./${filename:0:-4}

	############################################################
    
	for k in "${local_window_size_list[@]}"; do

		sed -e "s/CHANGE_filename/$filename/g" -e "s/CHANGE_local_window_size/$k/g" OrientationCoherance2D.py > OrientationCoherance2D_editedW$k.py

        ############################################################

        python3 OrientationCoherance2D_editedW$k.py
        
        rm OrientationCoherance2D_editedW$k.py

        ############################################################

		echo "Window size processed is $k"

		############################################################

	done # endloop for k

	echo ""

	echo "================================"

	cd ..

	############################################################

done
