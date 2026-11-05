"""Prevent repeated tray layouts leaking into ACT validation."""
import unittest
from tools.prepare_act_split import split_episodes


class ACTSplitTests(unittest.TestCase):
    def test_layout_groups_stay_together_and_failures_are_excluded(self):
        episodes=[]
        for worker in range(2):
            for local in range(20):
                episodes.append(dict(index=len(episodes),worker_id=worker,
                    worker_episode_index=local,complete=True,success=local%4!=0,frames=300))
        episodes[1]['complete']=False
        episodes[2]['frames']=0
        manifest=dict(tray_every=5,episodes=episodes)
        split=split_episodes(manifest,.25,42)
        self.assertEqual(split,split_episodes(manifest,.25,42))
        train=set(split['train_indices']);validation=set(split['validation_indices'])
        eligible={e['index'] for e in episodes if e['success'] and e['complete'] and e['frames']}
        self.assertEqual(train|validation,eligible)
        self.assertFalse(train&validation)
        groups=lambda indices:{(episodes[i]['worker_id'],episodes[i]['worker_episode_index']//5) for i in indices}
        self.assertFalse(groups(train)&groups(validation))
        self.assertEqual(len(groups(validation)),2)

    def test_one_layout_cannot_be_an_independent_validation_set(self):
        with self.assertRaisesRegex(ValueError,'two successful'):
            split_episodes(dict(tray_every=5,episodes=[dict(index=0,complete=True,success=True,frames=1)]))


if __name__=='__main__':unittest.main()
