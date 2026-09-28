import numpy as np
from scripts.spec_k_question_answer_mi import calculate

def test_constant_distribution_has_zero_question_answer_mi():
    p=np.tile(np.array([[.1,.2,.3,.4]]),(20,1)); r=calculate(p)
    assert abs(r["i_x_yhat_nats"])<1e-14
    assert r["sanity"]["route_agreement_10_significant_figures_pass"]

def test_deterministic_balanced_answers_reach_ceiling():
    p=np.eye(4).repeat(10,axis=0); r=calculate(p)
    assert r["i_x_yhat_bits"]==2.0
    assert r["fraction_of_two_bit_ceiling"]==1.0
    assert r["fraction_max_probability_gt_0_99"]==1.0

def test_entropy_and_kl_routes_agree():
    rng=np.random.default_rng(4); p=rng.dirichlet(np.ones(4),size=100); r=calculate(p)
    assert r["crosscheck_absolute_discrepancy_nats"]<1e-12
    assert all(r["sanity"].values())
