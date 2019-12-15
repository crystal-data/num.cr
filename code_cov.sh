echo "require \"./spec/**\"" > run_tests.cr && \
crystal build run_tests.cr -D skip-integration && \
kcov --include-path=$(pwd)/src $(pwd)/coverage ./run_tests && \
bash <(curl -s https://codecov.io/bash) -s $(pwd)/coverage
