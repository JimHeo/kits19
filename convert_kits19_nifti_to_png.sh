# python starter_code/visualize.py -c case_00123 -d ./data_png -o False

CASE_SET=$(seq -f %05g 0 209)
OVERLAY=False
PLANE="axial"

CASE=0
for CASE_00 in $CASE_SET; do
    DEST="./data_png/case_"${CASE_00}
    echo "Converting to png files case_"${CASE_00}" to "${DEST}"..."
    DEST="./data_png/case_"${CASE_00}"/"${PLANE}
    echo "case_"${CASE_00}" - "${PLANE}
    python starter_code/visualize.py -c ${CASE} -d ${DEST} -p ${PLANE} -o ${OVERLAY}
    CASE=$(($CASE+1))
done

# CASE=0
# for CASE_00 in $CASE_SET; do
#     DEST="./data_png/case_"${CASE_00}
#     echo "Converting to png files case_"${CASE_00}" to "${DEST}"..."
#     for PLANE in "axial" "coronal" "sagittal"; do
#         DEST="./data_png/case_"${CASE_00}"/"${PLANE}
#         echo "case_"${CASE_00}" - "${PLANE}
#         python starter_code/visualize.py -c ${CASE} -d ${DEST} -p ${PLANE} -o ${OVERLAY}
#         CASE=$(($CASE+1))
#     done
# done