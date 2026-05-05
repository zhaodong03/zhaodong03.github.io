export interface TimelineEntry {
  title: string;
  institution: string;
  location?: string;
  period: string;
  description: string[];
  link?: string;
}

export interface Skill {
  category: string;
  items: string[];
}

export const education: TimelineEntry[] = [
  {
    title: 'M.S. in Computer Science',
    institution: 'Georgia Institute of Technology',
    location: 'Atlanta, GA',
    period: 'Currently Pursuing',
    description: [
      'GPA: 4.0/4.0',
      'Coursework: Operating Systems, GPU SW/HW, Big Data Health, Health Informatics',
    ],
    link: 'https://www.gatech.edu',
  },
  {
    title: 'B.S. in Computer Science — Intelligence + Information Internetwork',
    institution: 'Georgia Institute of Technology',
    location: 'Atlanta, GA',
    period: 'Aug 2021 – May 2024',
    description: [
      'GPA: 3.96/4.0 · Major GPA: 4.0/4.0',
      'Awards: Faculty Honors, Dean\'s List',
      'Engagements: CoC Peer Mentor, HackGT, GT Trailblazers, CSSA',
    ],
    link: 'https://www.gatech.edu',
  },
  {
    title: 'Early College Scholar',
    institution: 'Washington University in St. Louis',
    location: 'St. Louis, MO',
    period: 'Jul 2020',
    description: [
      'Grade: A+',
      'Course: Design Thinking — Human-Centered Approaches to Making the World (3 credits)',
    ],
    link: 'https://wustl.edu/',
  },
];

export const workExperience: TimelineEntry[] = [
  {
    title: 'Software Development Engineer — HW/SW Co-Design Chip Validation',
    institution: 'Annapurna Labs (AWS)',
    location: 'Austin, TX',
    period: 'Dec 2025 – Present',
    description: [
      'Designed and implemented robust modern C++ test benches for critical data movement architecture, validating DMA operations in silicon bring-up.',
    ],
  },
  {
    title: 'Software Engineer',
    institution: 'Bank of America',
    location: 'Charlotte, NC',
    period: 'Jul 2024 – Nov 2025',
    description: [
      'Optimized the FR Y-14Q Schedule L regulatory template download process, reducing execution time by 75% (~30 min → 7 min).',
      'Built a Python-based Data Quality Control system in Quartz to identify counterparty financial data inconsistencies in Enterprise Stress Testing, enabling automated reconciliation.',
      'Maintained 99% unit test coverage across all production releases; drove UAT and regression testing.',
    ],
  },
  {
    title: 'Software Engineer Intern — Global Technology',
    institution: 'Bank of America',
    location: 'Charlotte, NC',
    period: 'Jun 2023 – Aug 2023',
    description: [
      'Designed and built a Python-based Workflow Orchestration Tool automating 20+ pre-processing steps for regulatory reporting in Oracle DB, cutting manual setup time by 80%.',
      'Integrated real-time monitoring and status tracking into the internal UI framework, reducing user-induced errors via automated validation checks.',
    ],
  },
];

export const researchExperience: TimelineEntry[] = [
  {
    title: 'Undergraduate Research Assistant',
    institution: 'Ubicomp Health Lab @ Georgia Tech',
    location: 'Atlanta, GA',
    period: 'Aug 2022 – Jan 2023',
    description: [
      'Conducted research with Dr. Rosa Arriaga about Human-Computer Interaction in Ubicomp Health Lab, focusing on building the Diabetic Ulcer Computational Sensing System for diabetes patients and clinicians that is usable and useful.',
      'Working with a group of 9 on designing and developing the system with React Native, reviewing related literature, and participating in weekly meetings.',
    ],
    link: 'https://sites.google.com/view/riarriaga/lab',
  },
  {
    title: 'Automatic Algorithm Design Team Member (VIP)',
    institution: 'Georgia Tech Research Institute',
    location: 'Atlanta, GA',
    period: 'Jan 2022 – May 2023',
    description: [
      'Participated in bootcamp to learn Genetic Algorithms, Genetic Programming, Machine Learning, DEAP (Distributed Evolutionary Algorithms in Python), and EMADE (Evolutionary Multi-objective Algorithm Design Engine).',
      'Led a group of 4 to achieve co-dominant solutions and find the Pareto front using different ML models, Multi-Objective Genetic Programming, and EMADE on predicting Titanic survival — Accuracy 86.59%, True Positive Rate 88.03%.',
      'Joined image-processing sub-team working with 2 students to preprocess 49,466 images for testing the Evolutionary Model.',
      'Documented legible and technical meeting notes with MediaWiki to track lab progress.',
    ],
  },
];

export const experience = [...workExperience, ...researchExperience];

export const coursework = [
  {
    semester: 'Graduate',
    courses: ['Introduction to Operating Systems', 'GPU SW and HW', 'Big Data Health', 'Intro to Health Informatics'],
  },
  {
    semester: 'Spring 2024',
    courses: ['Computer Networks I', 'Automata and Complexity', 'Intro to Information Security'],
  },
  {
    semester: 'Fall 2023',
    courses: ['Introduction to Perception and Robotics', 'Statistics and Applications', 'Applied Combinatorics', 'Computing, Society, and Professionalism'],
  },
  {
    semester: 'Spring 2023',
    courses: ['Design & Analysis of Algorithms', 'Objects and Design', 'Introduction to Artificial Intelligence', 'General Psychology'],
  },
  {
    semester: 'Fall 2022',
    courses: ['Intro to Database Systems', 'Intro to Computer Vision', 'Machine Learning', 'Systems and Networks', 'Intro Multivariable Calculus'],
  },
  {
    semester: 'Spring 2022',
    courses: ['Data Structures & Algorithms', 'Computer Organization & Programming', 'Honors Discrete Math CS'],
  },
  {
    semester: 'Fall 2021',
    courses: ['Intro to Object Oriented Programming', 'Linear Algebra'],
  },
];

export const cvSkills: Skill[] = [
  {
    category: 'Languages',
    items: ['Python', 'Java', 'C', 'C++', 'TypeScript', 'JavaScript', 'SQL', 'MySQL'],
  },
  {
    category: 'Frameworks & Libraries',
    items: ['React', 'Next.js', 'React Native', 'PyTorch', 'gRPC', 'Protocol Buffers', 'MyBatis'],
  },
  {
    category: 'Tools & Platforms',
    items: ['Git', 'Docker', 'Linux', 'Oracle DB', 'SQLite', 'Streamlit', 'MediaWiki'],
  },
  {
    category: 'Concepts',
    items: [
      'Distributed Systems',
      'Machine Learning',
      'Computer Vision',
      'Database Systems',
      'HCI',
      'Systems Programming',
    ],
  },
];
