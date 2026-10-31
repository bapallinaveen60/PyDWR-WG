import unittest
import torch
from pydwr_wg.models.layers import CircularAzimuthPad2d, SeparableConv2d, AttentionGate
from pydwr_wg.models.pta_innet import PTAInNet

class TestLayersAndModel(unittest.TestCase):
    def test_circular_azimuth_pad(self):
        pad = CircularAzimuthPad2d(pad_range=1, pad_azimuth=2)
        x = torch.randn(2, 4, 10, 20)
        out = pad(x)
        self.assertEqual(out.shape, (2, 4, 12, 24))
        self.assertTrue(torch.allclose(out[:, :, 1:-1, :2], x[:, :, :, -2:]))

    def test_separable_conv(self):
        conv = SeparableConv2d(in_channels=16, out_channels=32, kernel_size=3)
        x = torch.randn(2, 16, 20, 36)
        out = conv(x)
        self.assertEqual(out.shape, (2, 32, 20, 36))

    def test_attention_gate(self):
        gate = AttentionGate(f_gate=32, f_x=64, f_int=16)
        g = torch.randn(2, 32, 10, 18)
        x = torch.randn(2, 64, 20, 36)
        out = gate(g, x)
        self.assertEqual(out.shape, (2, 64, 20, 36))

    def test_pta_innet_forward(self):
        model = PTAInNet(in_channels=10, encoder_channels=(16, 32, 64, 128, 256), num_groups=4)
        model.eval()

        x = torch.randn(1, 10, 64, 36)
        with torch.no_grad():
            preds = model(x)

        self.assertIn("clutter_mask", preds)
        self.assertIn("zh_clean", preds)
        self.assertIn("rain_rate", preds)

        self.assertEqual(preds["clutter_mask"].shape, (1, 1, 64, 36))
        self.assertEqual(preds["zh_clean"].shape, (1, 1, 64, 36))
        self.assertEqual(preds["rain_rate"].shape, (1, 1, 64, 36))

        self.assertTrue(torch.all(preds["clutter_mask"] >= 0.0))
        self.assertTrue(torch.all(preds["clutter_mask"] <= 1.0))
        self.assertTrue(torch.all(preds["rain_rate"] >= 0.0))

if __name__ == "__main__":
    unittest.main()
