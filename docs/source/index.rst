.. Catanatron documentation master file, created by
   sphinx-quickstart on Fri Jul  2 14:25:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. Helpful link: https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/
.. Helpful link 2: https://stackoverflow.com/questions/46278683/include-my-markdown-readme-into-sphinx

Welcome to Catanatron's documentation!
======================================

Catanatron is a Python implementation of the Settlers of Catan core game logic.

The most basic usage of the package looks like:

.. code-block::

   from catanatron.game import Game
   from catanatron.models.player import RandomPlayer, Color

   players = [
      RandomPlayer(Color.RED),
      RandomPlayer(Color.BLUE),
      RandomPlayer(Color.WHITE),
      RandomPlayer(Color.ORANGE),
   ]
   game = Game(players)
   game.play()  # returns winning color

The example above executes a game of 4 RandomPlayers (players that decide completly at random).
The ``catanatron.players`` has more ``Player`` implementations.

Although the package exposes an OOP API, internally we've been moving to a 
Functional implementation for performance. Particularly, when making copies of
the state (a common operation for tree-searching algorithms), it's much faster to 
copy a dictionary of primitives than to ``copy.deepcopy`` an entire class.

Internally the ``Game.play()`` method is mainly a while-no-one-has-won loop
that asks players to decide on a possible action. This architecture naturally
makes the framework fit the `Game Trees <https://en.wikipedia.org/wiki/Game_tree>`_ concept, and makes it easy to implement
tree-searching players.

Thus, Players must implement the following API:

.. code-block::

   from catanatron.game import Game
   from catanatron.models.actions import Action
   from catanatron.models.player import Player

   class MyPlayer(Player):
      def decide(self, game: Game, playable_actions: Iterable[Action]):
         """Should return one of the playable_actions.

         Args:
               game (Game): complete game state. read-only.
               playable_actions (Iterable[Action]): options to choose from
         Return:
               action (Action): Chosen element of playable_actions
         """
         raise NotImplementedError

The first parameter, ``game`` is mainly so that the player can access ``game.state``
to take its decisions. This ``game.state`` is currently represented by a simple data 
container class; you can see the documentation here: 
https://catanatron.readthedocs.io/en/latest/catanatron.html#catanatron.state.State.
For now, players should not mutate this state (should treat it read-only). If one
would like to make modifications to consider actions one should copy the state with
the ``State.copy`` function.

The second parameter is the list of playable Actions. An ``Action`` is a tuple of 
enums and primitives like: 

- ``(ActionType.PLAY_MONOPOLY, WHEAT)`` (i.e. play monopoly card and select wheat)
- ``(ActionType.BUILD_SETTLEMENT, 3)`` (i.e. build settlement on node 3)
- ``(ActionType.MOVE_ROBBER, (1,0,1), Color.BLUE)`` (i.e. move robber to tile on coordinate (1,0,1) and steal from blue)
- ``(ActionType.END_TURN, None)`` (i.e. do nothing else and end turn)

After a player takes a decision, the Game follows a Redux/Router pattern in which calls
generic ``apply_action(state, action)`` method that will route to the appropriate 
state-mutating function in the ``state_functions`` module.

A great way to further undersand the internals is to place a breakpoint in ``MyPlayer`` or ``RandomPlayer``
run the provided sample.py in the repo and inspect the ``game`` and ``playable_actions`` objects.

This package is published in PyPi and is pip-installable like so:

.. code-block::

   pip install catanatron

API Reference
=============

.. toctree::
   :maxdepth: 4

   catanatron
   catanatron.gym



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
