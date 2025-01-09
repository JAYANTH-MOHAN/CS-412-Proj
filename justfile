set shell := ["bash", "-uc"]
set positional-arguments
set dotenv-load

noti msg:
    curl -X POST -H "Content-type: application/json" -d "{'text': '$1'}" "https://hooks.slack.com/services/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"
bench_speed:
    python -m optillm.cot_decoding
eval dataset:
    # python -m optillm.eval_speculative_cot --dataset $1
    python -m optillm.eval_speculative_cot --dataset $1 --eval_only --speculative
    python -m optillm.eval_speculative_cot --dataset $1 --eval_only
    python -m optillm.eval_speculative_cot --dataset $1 --eval_only --greedy
    python -m optillm.eval_speculative_cot --dataset $1 --eval_only --self_consistency
install:
    pip install bitsandbytes accelerate contexttimer datasets flash-attn ipdb
login:
    pip install -U "huggingface_hub[cli]"
    huggingface-cli login
run_multiarith:
    python -m optillm.eval_speculative_cot --dataset multiarith --speculative
    # python -m optillm.eval_speculative_cot --dataset multiarith --greedy
    # python -m optillm.eval_speculative_cot --dataset multiarith
    # python -m optillm.eval_speculative_cot --dataset multiarith --self_consistency
