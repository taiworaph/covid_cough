#! /bin/bash


# clear the initial directory of all wav files before 
find . -type f -name '*.wav' -exec rm -f {} \;
echo "Removed all .wav files from this  directory prior to converting .webm files to .wav"

array_files=()
i=0
for webm in *.webm
do
	array_files=(${array_files[@]} "$webm")
	echo "This is $i"
	((i++))

done

echo ${#array_files[@]}
echo "${array_files[@]}"


#create a new directory
dirName="only_wave_files"
[ -e "$dirName" ] && echo $dirName exists || mkdir "$dirName"
#mkdir only_wave_files


# loop through the item and save them to the current directory with same filename
for item in ${array_files[*]}
do
	filename="${item%.*}"

	echo $filename
	#new_filename=$filename+".wav"
	ffmpeg -i $item -ac 2 -f wav $filename.wav
	new_path=$dirName+/+$filename.wav
	mv $new_filename $new_path
done

echo "--------------------------------------------"
echo "-----------------Completed------------------"
echo "--------------------------------------------"