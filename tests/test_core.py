import tempfile
import unittest
from pathlib import Path
import torch
from minimodel import ModelConfig, TransformerLM, Tokenizer
from minimodel.training import TrainConfig, train, load_checkpoint, evaluate

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tokenizers(self):
        for mode in ("char", "byte"):
            t = Tokenizer.fit("hello café 🌱", mode)
            self.assertEqual(t.decode(t.encode("hello café 🌱")), "hello café 🌱")
            self.assertEqual(Tokenizer(**t.state()).encode("hello"), t.encode("hello"))
        self.assertEqual(Tokenizer.fit("a").encode("b"), [0])

    def test_causality_and_gradients(self):
        torch.manual_seed(3)
        model = TransformerLM(ModelConfig(20, context=8, width=16, heads=4, layers=2, dropout=0))
        x = torch.randint(20, (2, 8))
        altered = x.clone()
        altered[:, 4:] = (altered[:, 4:] + 1) % 20
        self.assertTrue(torch.allclose(model(x)[0][:, :4], model(altered)[0][:, :4], atol=1e-6))
        model(x, x)[1].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertIs(model.head.weight, model.token.weight)

    def test_overfit_and_generation(self):
        torch.manual_seed(4)
        model = TransformerLM(ModelConfig(4, context=8, width=16, layers=1, dropout=0))
        x = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        y = (x + 1) % 4
        initial = model(x, y)[1].item()
        opt = torch.optim.AdamW(model.parameters(), lr=0.02)
        for _ in range(30):
            opt.zero_grad()
            loss = model(x, y)[1]
            loss.backward()
            opt.step()
        self.assertLess(model(x, y)[1].item(), initial * 0.4)
        result = model.generate(x, tokens=12, top_k=1)
        self.assertEqual(result.shape, (1, 20))
        self.assertTrue(model.training)

    def test_training_checkpoint_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "text.txt"
            data.write_text("the cat sat on the mat. " * 30)
            cfg = TrainConfig(data=str(data), output=str(Path(directory) / "run"), context=8, width=16, layers=1, steps=3, eval_interval=2, eval_batches=2, batch_size=2)
            history = train(cfg)
            self.assertEqual([r["step"] for r in history], [0, 2, 3])
            model, tokenizer, checkpoint = load_checkpoint(Path(cfg.output) / "last.pt")
            ids = torch.tensor(tokenizer.encode(data.read_text()))
            a, b = evaluate(model, ids, count=2), evaluate(model, ids, count=2)
            self.assertEqual(a, b)
            self.assertEqual(checkpoint["step"], 3)
            with self.assertRaises(ValueError):
                train(cfg)

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            ModelConfig(10, width=15, heads=4)
        with self.assertRaises(ValueError):
            TrainConfig(steps=0)
        model = TransformerLM(ModelConfig(10, context=4))
        with self.assertRaises(ValueError):
            model(torch.ones(1, 5, dtype=torch.long))

if __name__ == "__main__":
    unittest.main()
