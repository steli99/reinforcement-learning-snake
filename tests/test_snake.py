import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from environments_fully_observable import OriginalSnakeEnvironment as Full
from environments_partially_observable import OriginalSnakeEnvironment as Partial


def configure(env,head,body,fruit):
    board=env.boards[0]
    board[1:-1,1:-1]=env.EMPTY
    env.bodies[0]=list(body)
    for cell in body:board[cell]=env.BODY
    board[head]=env.HEAD
    board[fruit]=env.FRUIT


class EnvironmentTests(unittest.TestCase):
    def test_seed_and_observation(self):
        a,b=Full(4,7,seed=5),Full(4,7,seed=5)
        np.testing.assert_array_equal(a.boards,b.boards)
        self.assertEqual(a.to_state().shape,(4,7,7,4))
        self.assertEqual(a.to_state().dtype,np.float32)
        self.assertTrue(np.all((a.boards==a.HEAD).sum((1,2))==1))
        self.assertTrue(np.all((a.boards==a.FRUIT).sum((1,2))==1))

    def test_fruit_grows_body(self):
        env=Full(1,5,seed=0);configure(env,(2,2),[],(2,3))
        _,r,d,info=env.step([env.RIGHT])
        self.assertAlmostEqual(float(r[0]),.5)
        self.assertEqual(env.bodies[0],[(2,2)])
        self.assertFalse(d[0]);self.assertTrue(info['fruit'][0])
        self.assertEqual(np.sum(env.boards==env.FRUIT),1)

    def test_wall_does_not_corrupt_body(self):
        env=Full(1,5,seed=0);configure(env,(1,1),[(1,2),(2,2)],(3,3))
        before=env.boards.copy();body=list(env.bodies[0])
        _,r,d,info=env.step([[env.DOWN]])
        np.testing.assert_array_equal(before,env.boards)
        self.assertEqual(body,env.bodies[0]);self.assertFalse(d[0])
        self.assertAlmostEqual(float(r[0]),-.1);self.assertTrue(info['wall'][0])

    def test_self_bite_truncates_without_termination(self):
        env=Full(1,5,seed=0);configure(env,(2,2),[(2,1),(3,1),(3,2)],(1,1))
        _,r,d,info=env.step([env.LEFT])
        self.assertAlmostEqual(float(r[0]),-.2)
        self.assertEqual(env.bodies[0],[]);self.assertFalse(d[0]);self.assertTrue(info['self_bite'][0])

    def test_win_resets_and_marks_terminal(self):
        env=Full(1,4,seed=0);configure(env,(1,1),[(2,1),(2,2)],(1,2))
        _,r,d,info=env.step([env.RIGHT])
        self.assertEqual(float(r[0]),1.);self.assertTrue(d[0]);self.assertTrue(info['win'][0])
        self.assertEqual(env.bodies[0],[])
        self.assertEqual(np.sum(env.boards==env.HEAD),1)
        self.assertEqual(np.sum(env.boards==env.FRUIT),1)

    def test_partial_view_matches_full_crop_and_dynamics(self):
        full,partial=Full(2,7,seed=1),Partial(2,7,2,seed=1)
        for _ in range(12):
            np.testing.assert_array_equal(full.boards,partial.boards)
            obs=partial.to_state();self.assertEqual(obs.shape,(2,5,5,4))
            self.assertTrue(np.all(obs[:,2,2,3]==1))
            a=np.array([0,1]);full.step(a);partial.step(a)
        zero=Partial(1,7,0,seed=0)
        self.assertEqual(zero.to_state().shape,(1,1,1,4))

    def test_invalid_actions_and_input_immutability(self):
        env=Full(2,7,seed=1)
        for actions in [[4,0],[1.5,0],[0],[np.nan,0]]:
            with self.assertRaises(ValueError):env.step(actions)
        actions=np.array([[0],[1]]);old=actions.copy();env.move(actions)
        np.testing.assert_array_equal(actions,old)

    def test_invariants_after_random_play(self):
        env=Full(8,7,seed=9);rng=np.random.default_rng(0)
        for _ in range(500):
            env.step(rng.integers(0,4,8))
            for b,body in enumerate(env.bodies):
                self.assertEqual(len(body),len(set(body)))
                self.assertEqual(len(body),np.sum(env.boards[b]==env.BODY))
                for point in body:self.assertEqual(env.boards[b][point],env.BODY)
            self.assertTrue(np.all((env.boards==env.HEAD).sum((1,2))==1))
            self.assertTrue(np.all((env.boards==env.FRUIT).sum((1,2))==1))


class LearningTests(unittest.TestCase):
    def test_reset_bootstrap_is_masked(self):
        from rl_agent import discounted_returns
        # The last transition in board 0 resets; board 1 must bootstrap.
        actual=discounted_returns([[1,1],[2,2]],[[False,False],[True,False]],[100,10],.9)
        np.testing.assert_allclose(actual,[[2.8,10.9],[2,11]],rtol=1e-6)

    def test_finite_gradient_and_weight_update(self):
        from rl_agent import A2C,set_seed
        import tensorflow as tf
        set_seed(0);env=Full(8,7,seed=2);learner=A2C(env.to_state().shape[1:])
        state=tf.constant(env.to_state());actions,_=learner.sample(state)
        before=[v.numpy().copy() for v in learner.model.trainable_variables]
        values=learner.update(state,actions,tf.constant(np.arange(8,dtype=np.float32)/8))
        self.assertTrue(all(np.isfinite(v.numpy()) for v in values))
        self.assertTrue(any(not np.array_equal(a,b.numpy()) for a,b in zip(before,learner.model.trainable_variables)))

if __name__=='__main__':unittest.main()
