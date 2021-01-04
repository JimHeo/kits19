# python starter_code/visualize.py -c case_00123 -d ./data_png -o False

DEST="./ex"
CASE_SET=$(seq 0 299)
OVERLAY=True

for CASE in $CASE_SET; do
    echo "Converting to png files case_00"${CASE}" to "${DEST}"..."
    for PLANE in "axial" "coronal" "sagittal"; do
        echo "case_00"${CASE}" - "${PLANE}
        python starter_code/visualize.py -c ${CASE} -d ${DEST} -o ${OVERLAY} -p ${PLANE}
    done
done
