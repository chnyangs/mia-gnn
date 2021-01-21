# shellcheck disable=SC2034
# shellcheck disable=SC2068
# define run function
run() {
    number=$1
    shift
    for i in $(seq $number); do
      $@
    done
}
echo $2
# $1 defines the number will be repeat
run "$1" python ../main.py --config '../'"$2";

