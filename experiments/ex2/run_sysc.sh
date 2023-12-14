
export LD_LIBRARY_PATH=/working_dir/experiments/ex2/ex2_sysc:/working_dir/builds/llvm-project/build-x86/lib

ulimit -s 65536

APPDIR=ex2_sysc


source ./$APPDIR/appslist.sh

for INPUT in ${AppArray[@]}; do
    if test -f "$APPDIR/$INPUT"; then
        echo "Running $INPUT ..."
        $APPDIR/$INPUT
        echo "... finished $INPUT"
        sleep 0.1
    else
        echo "WARNING: File $INPUT does not exist"
    fi
done # End of INPUT loop