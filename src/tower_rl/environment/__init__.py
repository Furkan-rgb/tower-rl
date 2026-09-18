"""What a run of the game is, as the learner sees it.

The actions a run admits, the state one exact reading normalises to, the reward
and episode vocabulary, the port one instrumented instance is driven through,
the environment that turns a decision into a transition, and the profile that
records where a decision's time went.

This is the root package: it imports nothing else from `tower_rl`. Everything
that simulates a run, learns from one, or measures one depends on it, and it
depends on none of them.
"""
