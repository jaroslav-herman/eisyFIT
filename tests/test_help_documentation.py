import unittest
from pathlib import Path

import eis_gui


class HelpDocumentationTests(unittest.TestCase):
    def test_help_docs_directory_points_to_bundled_markdown(self):
        docs = eis_gui._help_docs_directory()
        self.assertTrue(docs.is_dir())
        self.assertTrue((docs / "quick-start.md").is_file())

    def test_every_help_topic_exists_and_has_a_unique_menu_label(self):
        labels = [label for label, _filename, _title in eis_gui.HELP_TOPICS]
        filenames = [filename for _label, filename, _title in eis_gui.HELP_TOPICS]
        titles = [title for _label, _filename, title in eis_gui.HELP_TOPICS]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(filenames), len(set(filenames)))
        self.assertEqual(len(titles), len(set(titles)))
        for _label, filename, title in eis_gui.HELP_TOPICS:
            path = eis_gui._help_document_path(filename)
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.read_text(encoding="utf-8").strip())
            self.assertEqual(eis_gui._help_topic_label(filename), title)

    def test_help_menu_source_lists_all_documented_topics(self):
        source = Path(eis_gui.__file__).read_text(encoding="utf-8")
        self.assertIn('menu_bar.add_cascade(label="Help", menu=self.help_menu)', source)
        self.assertIn('label="Open documentation file\\u2026"', source)
        self.assertIn('label="About eisyFIT"', source)
        self.assertIn('self.root.bind("<F1>"', source)


if __name__ == "__main__":
    unittest.main()
