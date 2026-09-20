"""Regression checks for grouping, quarantine, and decision-time boundaries."""

import unittest

from scripts.prepare_chat_pool import (
    Groups, exact_keys, jaccard, normalize, prefix_indices, privacy_flags, valid_messages,
)


def user(text):
    return {"role": "user", "content": text}


class PoolTests(unittest.TestCase):
    def test_quarantine_propagates_through_transitive_union(self):
        groups = Groups()
        left, middle, explored = groups.add(), groups.add(), groups.add(True)
        groups.union(middle, explored)
        groups.union(left, middle)
        self.assertEqual(groups.find(left), groups.find(explored))
        self.assertTrue(groups.explored[groups.find(left)])

    def test_shared_document_with_different_questions_stays_together(self):
        document = "这是需要共同隔离的原始材料。" * 20
        left = exact_keys([user("阅读材料\n" + document + "\n问题：有哪些人物？")])
        right = exact_keys([user("阅读材料\n" + document + "\n问题：事件发生在哪里？")])
        self.assertTrue(left & right)

    def test_common_short_greeting_does_not_merge_distinct_conversations(self):
        self.assertFalse(exact_keys([user("你好"), user("如何调试网络")]) &
                         exact_keys([user("你好"), user("如何做饭")]))

    def test_normalization_is_stable_and_shared_prompt_branches_group(self):
        prompt = "ＡＢＣ  内容 " * 30
        self.assertEqual(normalize(prompt), normalize(normalize(prompt)))
        self.assertTrue(exact_keys([user(prompt), user("详细解释")]) &
                        exact_keys([user(normalize(prompt)), user("换个角度")]))

    def test_prefix_stops_at_user_and_respects_whole_message_budget(self):
        messages = [user("ab"), {"role": "assistant", "content": "cde"}, user("fg"),
                    {"role": "assistant", "content": "future answer"}, user("late")]
        self.assertEqual(prefix_indices(messages, 7), [0, 2])
        self.assertEqual(messages[:prefix_indices(messages, 7)[-1]+1][-1], user("fg"))
        self.assertEqual(prefix_indices(messages, 6), [0])
        self.assertEqual(prefix_indices(messages, 1), [])

    def test_credentials_in_future_reply_also_block_source_sampling(self):
        messages = [user("help"), {"role": "assistant", "content": "sk-EXAMPLEONLY123456"}]
        self.assertIn("credential_pattern", privacy_flags(messages))

    def test_unsupported_roles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported_role"):
            valid_messages({"messages": [user("hi"), {"role": "tool", "content": "result"}]})

    def test_similarity_uses_union_not_only_containment(self):
        self.assertEqual(jaccard({1, 2}, {1, 2, 3, 4}), .5)
        self.assertEqual(jaccard({1, 2}, {3, 4}), 0)


if __name__ == "__main__":
    unittest.main()
