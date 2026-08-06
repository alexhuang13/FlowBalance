import torch
from recipe.rlsd.rlsd_core_algos import build_rlsd_advantages, resolve_lambda


def test_positive_adv_prefers_teacher_likely_tokens():
    teacher=torch.tensor([[0.0, -2.0]])
    student=torch.tensor([[-1.0, -1.0]])
    adv=torch.ones(1,2)
    mask=torch.ones(1,2)
    out,_=build_rlsd_advantages(teacher_log_probs=teacher,student_log_probs=student,advantages=adv,response_mask=mask,self_distillation_mask=torch.ones(1),lam=1.0,clip_range=0.2)
    assert torch.allclose(out,torch.tensor([[1.2,0.8]]))


def test_negative_adv_reverses_direction():
    teacher=torch.tensor([[0.0]])
    student=torch.tensor([[-1.0]])
    out,_=build_rlsd_advantages(teacher_log_probs=teacher,student_log_probs=student,advantages=-torch.ones(1,1),response_mask=torch.ones(1,1),self_distillation_mask=torch.ones(1),lam=1.0,clip_range=0.2)
    assert torch.allclose(out,torch.tensor([[-0.8]]))


def test_no_teacher_sample_is_plain_grpo():
    adv=torch.tensor([[2.0, -3.0]])
    out,_=build_rlsd_advantages(teacher_log_probs=torch.zeros_like(adv),student_log_probs=torch.ones_like(adv),advantages=adv,response_mask=torch.ones_like(adv),self_distillation_mask=torch.zeros(1),lam=1.0,clip_range=0.2)
    assert torch.equal(out,adv)


def test_lambda_schedule():
    assert resolve_lambda(0.5,5,10,20)==0.25
    assert resolve_lambda(0.5,10,10,20)==0.5
    assert resolve_lambda(0.5,20,10,20)==0.25
    assert resolve_lambda(0.5,30,10,20)==0.0
