import unittest
from utils import TrainingConfigure
from visual import visualize_model, visualize_model_single_label

class VisualTest(unittest.TestCase):
    # def test_first(self):
    #     config = TrainingConfigure.from_file('example.json')
    #     visualize_model('logdir/default_best(66.08%)_Seq2SeqGRU_2024-03-27_21-54-34.pt', config)
    #     self.assertEqual(True, True)  # add assertion here
    #
    # def test_visual_single_label(self):
    #     config = TrainingConfigure.from_file('example.json')
    #     visualize_model_single_label('logdir/default_best(66.08%)_Seq2SeqGRU_2024-03-27_21-54-34.pt', config)
    #     self.assertEqual(True, True)  # add assertion here

    def test(self):
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
