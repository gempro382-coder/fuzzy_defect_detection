#!/usr/bin/env bash
# ===========================================================================
#  NeuraFuzz Inspect — hybrid ML + Fuzzy defect detection launcher
#
#    ./run.sh install     install python dependencies
#    ./run.sh dataset     generate the 1000-sample labelled dataset
#    ./run.sh train       train the ML model (10-stage pipeline)
#    ./run.sh evaluate    evaluate the fused system (ML vs Fuzzy vs Fused)
#    ./run.sh preview     rebuild the offline static dashboard snapshot
#    ./run.sh web         launch the production web UI  (default)
#    ./run.sh all         dataset -> train -> evaluate -> web
# ===========================================================================
set -e
cd "$(dirname "$0")"
CMD="${1:-web}"

case "$CMD" in
  install)  pip install -r requirements.txt ;;
  dataset)  (cd src && python dataset_generator.py --n 1000 --img-size 256 --seed 42) ;;
  train)    (cd src && python train_model.py) ;;
  evaluate) (cd src && python evaluate_system.py) ;;
  preview)  (cd web && python build_static_preview.py) ;;
  web)
    echo "Starting NeuraFuzz web UI on http://127.0.0.1:5000  (Ctrl+C to stop)"
    (cd web && python app.py) ;;
  all)
    (cd src && python dataset_generator.py --n 1000 --img-size 256 --seed 42)
    (cd src && python train_model.py)
    (cd src && python evaluate_system.py)
    echo "Now run: ./run.sh web" ;;
  *)
    echo "Usage: ./run.sh [install|dataset|train|evaluate|preview|web|all]"; exit 1 ;;
esac
