./generate_all.py --template srcs/template_mlir_matmuls.mlir > srcs/mlir_matmuls.mlir
./generate_all.py --template srcs/template_mlir_matmuls.h > srcs/mlir_matmuls.h.inc
./generate_all.py --template srcs/template_cmd.h > scripts/compile_mlir_matmul-all.sh