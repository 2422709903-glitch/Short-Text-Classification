import transformers.masking_utils as masking_utils


def xpu_safe_padding_mask_function(padding_mask):

    def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
        return padding_mask[:, None, None, :]

    return inner_mask


masking_utils.padding_mask_function = xpu_safe_padding_mask_function

print("XPU Padding Mask 广播修复已启用！")
