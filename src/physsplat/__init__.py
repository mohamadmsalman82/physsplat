"""PhysSplat: photo of simple objects in, interactive neural physics out.

The pipeline, and which subpackage owns each piece:

    photo
      |  recon/     lift to 3D Gaussians, split into objects, sample particles
      v
    scene packet (Gaussians + physics particles + body metadata)
      |  model/     the learned simulator: graph -> GNN -> rigid integration
      v
    motion, 60x per second, eventually in the browser (web/ + export/)

The model is trained beforehand on synthetic data:

    datagen/   PyBullet generates ground-truth trajectories (the ONLY training data)
    train/     dataset loading + the training loop
    eval/      rollout metrics and side-by-side videos vs. PyBullet

    common/    constants and helpers shared by BOTH datagen and recon --
               anything that must match exactly between training data and
               real reconstructed scenes lives here, on purpose.

Build order follows the tutorial phases: datagen -> model -> train -> eval,
then recon, then export/web. Later packages import earlier ones, never the
reverse.
"""


def main() -> None:
    print("Hello from physsplat!")
