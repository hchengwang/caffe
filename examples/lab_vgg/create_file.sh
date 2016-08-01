DATA=data/lab_dataset
MY=examples/lab

echo "Create train.txt...."
rm -rf $MY/train.txt
for i in 0 1 2 3
do
find $DATA/train -name $i*.jpg | cut -d '/' -f4-5 | sed "s/$/ $i/">>$MY/train.txt
done

echo "Create test.txt...."
rm -rf $MY/test.txt
for i in 0 1 2 3
do
find $DATA/test -name $i*.jpg | cut -d '/' -f4-5 | sed "s/$/ $i/">>$MY/test.txt
done
echo "All done"

