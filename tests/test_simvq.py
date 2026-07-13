import unittest

import torch
import torch.nn.functional as F

from encoder.quantization.simvq_moe import SimVQ1D


class SimVQ1DTest(unittest.TestCase):
    def test_decode_uses_projected_codebook(self):
        quantizer = SimVQ1D(n_e=4, e_dim=3)
        with torch.no_grad():
            quantizer.embedding.weight.copy_(torch.arange(12).reshape(4, 3))
            quantizer.embedding_proj.weight.copy_(2 * torch.eye(3))
            quantizer.embedding_proj.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))

        indices = torch.tensor([[[0, 2], [1, 3]]])

        decoded = quantizer.decode(indices)

        codebook = quantizer.embedding_proj(quantizer.embedding.weight)
        expected = F.embedding(indices.squeeze(0), codebook).transpose(1, 2)
        torch.testing.assert_close(decoded, expected)


if __name__ == "__main__":
    unittest.main()
