"""Phase 2: the learned simulator itself (tutorial Steps 17-28).

One physics step = build graph -> encode -> message passing -> decode ->
integrate. Running that in a loop on its own output is a "rollout".

Files:
    graph.py       every step, connect particles closer than CONTACT_RADIUS
                   into a graph (grid-hash neighbor search, O(N)). Edges
                   between different bodies are where contact forces live.
    gnn.py         Encode-Process-Decode network, plain PyTorch (no PyG:
                   better MPS support, clean ONNX export).
                   - encoder: small MLPs lift node/edge features to 128-dim
                   - processor: 10 message-passing blocks with residuals
                   - decoder: pools per BODY and outputs one linear + one
                     angular acceleration (6 numbers per object), so every
                     particle moves under one rigid transform and objects
                     can never "melt". A per-particle + Kabsch-projection
                     fallback head shares the same encoder/processor.
    integrator.py  semi-implicit Euler on SE(3): gravity added analytically
                   (the net learns only contact/friction residuals), then
                   velocity update -> position/rotation update -> particles
                   from body poses. Kept a pure function: Phase 7 re-implements
                   it in TypeScript and purity makes the two comparable.

Key invariant: the network never sees absolute positions, only relative
displacements and velocities -- translation invariance for free.
"""
