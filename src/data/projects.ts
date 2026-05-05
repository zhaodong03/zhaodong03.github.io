export interface Project {
  title: string;
  year: string;
  category: 'systems' | 'ml' | 'web' | 'research';
  description: string[];
  tags: string[];
  link?: string;
}

export const projects: Project[] = [
  {
    title: 'Closed Captions Simplifier with Fine-tuned LLM',
    year: '2025',
    category: 'ml',
    description: [
      'Built and deployed an end-to-end ML system fine-tuning LLMs via LoRA for abstractive closed caption simplification — 34% length reduction, >90% content retention.',
      'Integrated Whisper ASR and built a real-time inference pipeline and demo app for latency-aware NLP.',
    ],
    tags: ['Python', 'PyTorch', 'LoRA', 'Whisper', 'NLP'],
  },
  {
    title: 'Distributed File System',
    year: '2025',
    category: 'systems',
    description: [
      'Designed and implemented a high-performance distributed file system with gRPC and Protocol Buffers, supporting scalable multi-client access with weak consistency.',
      'Built an efficient in-memory caching proxy with multi-threading and IPC primitives (semaphores, mutexes).',
    ],
    tags: ['C++', 'gRPC', 'Protocol Buffers', 'Multi-threading', 'IPC'],
  },
  {
    title: 'Acronym Search Engine',
    year: '2023',
    category: 'web',
    description: [
      'Led an 8-member team to build a scalable acronym search engine covering 30,000+ acronyms with Python, SQLite, and Streamlit.',
      'Significantly reduced internal training time and cost for the client organization.',
    ],
    tags: ['Python', 'SQLite', 'Streamlit', 'Team Lead'],
  },
  {
    title: 'Semantic Segmentation with PSPNet',
    year: '2022',
    category: 'ml',
    description: [
      'Developed a deep learning model for semantic segmentation using PSPNet architecture with ResNet backbone.',
      'Optimized via dilation and Pyramid Pooling Module (PPM); achieved mIoU of 60.4% on the CamVid dataset.',
    ],
    tags: ['Python', 'PyTorch', 'PSPNet', 'Computer Vision', 'Deep Learning'],
  },
  {
    title: 'Restaurant Supply Express',
    year: '2022',
    category: 'web',
    description: [
      'Collaborated with a team of 4 to design an EERD, create the physical DB schema, and implement views and stored procedures in MySQL.',
      'Implemented the backend with MyBatis, POJO, and REST API for a full-stack web application.',
    ],
    tags: ['Java', 'MySQL', 'MyBatis', 'REST API'],
    link: 'https://github.com/haozihong/GT-CS-4400-Project-Phase4',
  },
  {
    title: 'Facial Emotion Recognition',
    year: '2022',
    category: 'ml',
    description: [
      'Built a machine learning pipeline for real-time facial emotion classification.',
      'Evaluated multiple supervised and unsupervised approaches on standard benchmark datasets.',
    ],
    tags: ['Python', 'Machine Learning', 'Computer Vision', 'Scikit-learn'],
    link: 'https://github.gatech.edu/pages/Zkang40/CS-4641-Group-53',
  },
  {
    title: 'HackGT 8 — Hackathon Project',
    year: '2021',
    category: 'web',
    description: [
      'Built a project during the HackGT 8 hackathon at Georgia Tech.',
    ],
    tags: ['Hackathon', 'Georgia Tech'],
  },
  {
    title: 'Scene Recognition',
    year: '2022',
    category: 'ml',
    description: [
      'Implemented a scene recognition system using classical and deep learning computer vision techniques including SIFT features and RANSAC.',
    ],
    tags: ['Python', 'Computer Vision', 'SIFT', 'RANSAC'],
  },
  {
    title: 'Convolution & Image Processing',
    year: '2022',
    category: 'ml',
    description: [
      'Implemented convolution operations and image processing algorithms from scratch, exploring filtering, edge detection, and hybrid images.',
    ],
    tags: ['Python', 'NumPy', 'Computer Vision', 'Signal Processing'],
  },
];
