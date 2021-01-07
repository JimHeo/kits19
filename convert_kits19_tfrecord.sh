echo 'Tensorflow version 2.0.0 <= Required'

DEST="./tfrecord"
CASE_SET=$(seq -f %05g 0 209)
# CASE_SET=$(seq -f %05g 0 299)

CASE=0
for CASE_00 in $CASE_SET; do
    echo "Converting to tfrecord case_"${CASE_00}" to "${DEST}"..."
    python build_data.py --case ${CASE} --output-path ${DEST}"/kits19_case_"${CASE_00}"_train.tfrecord"
    CASE=$(($CASE+1))
done