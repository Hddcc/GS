import csv
import json
import tempfile
import unittest
from pathlib import Path

from audit_face_formal_paired import ALL_SCALES, main


FIELDS = ('method', 'dataset', 'image', 'scale', 'crop_border',
          'psnr_y', 'ssim_y', 'lpips')


def write_csv(path, dataset, offset=0, duplicate=False, omit=None):
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for scale in ALL_SCALES:
            for name in ('a.png', 'b.png', 'c.png'):
                if (scale, name) == omit:
                    continue
                row = {'method': 'model', 'dataset': dataset, 'image': name,
                       'scale': str(scale), 'crop_border': str(__import__('math').ceil(scale)),
                       'psnr_y': str(30 + offset), 'ssim_y': str(.8 + offset / 100),
                       'lpips': str(.2 - offset / 100)}
                writer.writerow(row)
                if duplicate and scale == 2 and name == 'a.png':
                    writer.writerow(row)


class FormalAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {}
        for dataset in ('CelebA', 'Helen'):
            for role, offset in (('baseline', 0), ('candidate', .05)):
                path = self.root / '{}_{}_per_image.csv'.format(role, dataset)
                write_csv(path, dataset, offset)
                self.paths[role, dataset] = path
        self.output = self.root / 'new_report.json'

    def args(self):
        return ['--metrics-dir', str(self.root), '--celeba-count', '3',
                '--helen-count', '3', '--draws', '200', '--output', str(self.output)]

    def test_complete_report_and_nonoverwrite(self):
        main(self.args())
        report = json.loads(self.output.read_text(encoding='utf-8'))
        self.assertEqual(list(report['results']), ['CelebA', 'Helen'])
        self.assertEqual(report['focus_scales'], [2., 2.5, 3.5, 4.])
        for result in report['results'].values():
            for metric, expected in (('psnr_y', .05), ('ssim_y', .0005), ('lpips', -.0005)):
                focus = result['focus_x2_to_x4'][metric]
                self.assertAlmostEqual(focus['mean'], expected)
                self.assertAlmostEqual(focus['ci95_image_bootstrap'][0], expected)
                self.assertEqual(focus['improved_fraction'], 1)
            self.assertEqual(len(result['scales']), len(ALL_SCALES))
        with self.assertRaises(FileExistsError):
            main(self.args())

    def test_duplicate_rejected(self):
        write_csv(self.paths['candidate', 'Helen'], 'Helen', .05, duplicate=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            main(self.args())
        self.assertFalse(self.output.exists())

    def test_unpaired_image_rejected(self):
        write_csv(self.paths['candidate', 'CelebA'], 'CelebA', .05,
                  omit=(2., 'b.png'))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            main(self.args())
        self.assertFalse(self.output.exists())

    def test_dataset_label_rejected(self):
        write_csv(self.paths['candidate', 'Helen'], 'CelebA', .05)
        with self.assertRaisesRegex(ValueError, 'Method/dataset mismatch'):
            main(self.args())

    def test_ambiguous_filename_requires_explicit_mapping(self):
        other = self.root / 'candidate_Helen_retry_per_image.csv'
        write_csv(other, 'Helen', .05)
        with self.assertRaisesRegex(ValueError, 'Expected exactly one'):
            main(self.args())


if __name__ == '__main__':
    unittest.main()
