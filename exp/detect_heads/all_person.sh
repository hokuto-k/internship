for i in `seq 1 30`
do
	mkdir result_image/${i}r
	python detect_head_from_image.py ${i} TownCentre-groundtruth.top
	ffmpeg -r 13 -i result_image/${i}r/testu_${i}_%04d.jpg -y result_movie/resultu_${i}.mp4
done