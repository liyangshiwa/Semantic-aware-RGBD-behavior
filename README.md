# Semantic-aware RGB-D network for classroom behavior detection


##  Introduction
Classroom behavior recognition is vital for analyzing student engagement and teaching quality. However, severe spatial occlusion in dense classrooms and visual ambiguities among similar behaviors (e.g., reading and writing) pose significant challenges for conventional single-modal 2D vision models. 
To tackle these limitations, this paper develops a semantic-aware RGB-D framework leveraging spatial geometric depth cues to resolve target occlusion and category confusion. The key contributions are as follows:
• Y-shaped Fusion Dual-stream Backbone (YFDB): Decouples shallow textural and geometric feature extraction while sharing deep semantic representations, significantly reducing architectural redundancy and computational cost.
• Cross-Scale Heterogeneous Context Aggregation (CSHCA): Dynamically fuses cross-modal features across stages, using spatial dynamic gating to suppress cluttered background noise and enhance salient action cues.
• Semantic Decoupling Head (SD-Head): Employs an asymmetric task decoupling design and multi-scale regional modeling to eliminate task interference between classification and regression.
• Superior Performance: Comprehensive evaluations on SCB-Dataset3, POPO, and SCB-Dataset show that our framework outperforms the baseline by 13.8%, 9.6%, and 11.8% in mAP50, respectively, verifying its robustness and real-time feasibility for smart education.



##  Environment & Requirements
We recommend running the codebase with **Python 3.10** and **PyTorch 2.2.1** with CUDA support.

# 1. Clone this repository
git clone [https://github.com/liyangshiwa/Semantic-aware-RGBD-behavior.git](https://github.com/liyangshiwa/Semantic-aware-RGBD-behavior.git)
cd Semantic-aware-RGBD-behavior

# 2. Create and activate a virtual environment
conda create -n rgbd_detect python=3.10 -y
conda activate rgbd_detect

# 3. Install core dependencies
pip install torch==2.2.1 torchvision==0.17.1 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install -r requirements.txt


## Code Availability & Status

> - Ongoing Cleanup: Due to GitHub file size constraints and ongoing code refactoring, large model checkpoint files (`.pt` / `.pth`) and complete end-to-end training pipelines are currently undergoing final cleanup and organization.
> - Full Release Commitment: The complete codebase, comprehensive training scripts, and full pre-trained weights will be made entirely open-source and publicly accessible upon formal acceptance of the paper.
> - Reviewer Access: If reviewers require temporary access to pre-trained weights for evaluation purposes during the peer-review process, please contact the corresponding author to request an evaluation package.
