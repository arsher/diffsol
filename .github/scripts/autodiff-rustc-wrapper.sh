#!/bin/sh

rustc="$1"
shift

case "$2" in
    diffsol|logistic_autodiff|closure_autodiff_rectangular)
        exec "$rustc" "$@" -Zautodiff=Enable
        ;;
esac

exec "$rustc" "$@"
