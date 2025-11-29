import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  Github,
  Brain,
  Shield,
  TrendingUp,
  Package,
  RotateCcw,
  MessageSquare,
  CheckCircle,
  ChevronRight,
  ExternalLink,
  ArrowRight,
  MessageCircle,
  Mail,
  Globe,
  X,
  Wrench,
  RefreshCcw,
  Lock,
  Eye,
  FileText,
  Check,
  User,
  Cpu,
  Zap,
} from "lucide-react";
import type { Contributor } from "@shared/schema";

interface GitHubStats {
  openIssues: number;
  contributors: number;
  mergedPRs: number;
  stars: number;
  forks: number;
}

interface GitHubIssue {
  title: string;
  bounty: string;
  skills: string[];
  difficulty: string;
  url: string;
}

interface HomePageProps {
  onNavigate: (sectionId: string) => void;
}

export default function HomePage({ onNavigate }: HomePageProps) {
  // Fetch live GitHub stats
  const { data: githubStats } = useQuery<GitHubStats>({
    queryKey: ["/api/github/stats"],
    refetchInterval: 60000,
  });

  // Fetch live GitHub issues with bounties
  const { data: githubIssues } = useQuery<GitHubIssue[]>({
    queryKey: ["/api/github/issues"],
    refetchInterval: 60000,
  });

  // Fetch live GitHub contributors
  const { data: contributors, isLoading: contributorsLoading } = useQuery<Contributor[]>({
    queryKey: ["/api/github/contributors"],
    refetchInterval: 60000, // Refresh every minute, but cached on server for 15 min
  });

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Hero Section */}
      <section
        id="home"
        className="min-h-screen flex items-center justify-center pt-20 px-4 relative overflow-hidden"
      >
        {/* Background gradient glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-radial from-blue-500/20 to-transparent rounded-full blur-3xl -z-10" />

        <div className="max-w-6xl mx-auto text-center">
          {/* Main Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-6xl sm:text-7xl lg:text-8xl font-extrabold leading-tight mb-6 bg-gradient-to-r from-gray-300 via-gray-200 to-blue-400 bg-clip-text text-transparent"
            data-testid="text-hero-headline"
          >
            THE AI-NATIVE
            <br />
            OPERATING SYSTEM
          </motion.h1>

          {/* Subheading */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="text-xl sm:text-2xl text-gray-400 max-w-3xl mx-auto mb-12 text-center"
            data-testid="text-hero-subheading"
          >
            Linux that understands natural language. No documentation required. Just ask, and
            Cortex executes.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="flex flex-col sm:flex-row gap-4 justify-center mb-16"
          >
            <button
              onClick={() => onNavigate("join")}
              className="px-8 py-4 bg-blue-500 rounded-lg text-lg font-semibold hover:shadow-[0_0_25px_rgba(59,130,246,0.5)] hover:scale-105 transition-all duration-300"
              data-testid="button-join-revolution"
            >
              Join the Revolution
            </button>
            <a
              href="https://github.com/cortexlinux/cortex"
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-4 border-2 border-blue-400 rounded-lg text-lg font-semibold hover:bg-blue-400/10 hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2"
              data-testid="link-github"
            >
              <Github size={20} />
              View on GitHub
            </a>
          </motion.div>

          {/* Stats Bar */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.8 }}
            className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-2xl mx-auto text-sm text-gray-400"
          >
            <div className="text-center" data-testid="stat-open-source">
              Open Source
            </div>
            <div className="text-center border-x border-white/10" data-testid="stat-community">
              Community Driven
            </div>
            <div className="text-center" data-testid="stat-market">
              $50-100B Market
            </div>
          </motion.div>
        </div>
      </section>

      {/* Statistics Section */}
      <section className="py-20 bg-gradient-to-b from-transparent to-black/50 border-t border-white/10">
        <div className="max-w-5xl mx-auto px-4">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-12" data-testid="text-project-momentum">
            Project Momentum
          </h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { 
                number: githubStats?.openIssues?.toString() || "29", 
                label: "Active Issues", 
                testid: "stat-active-issues" 
              },
              { 
                number: "$$$$$", 
                label: "In Bounties", 
                testid: "stat-bounties" 
              },
              { 
                number: githubStats?.contributors?.toString() || "8-10", 
                label: "Contributors", 
                testid: "stat-contributors" 
              },
              { 
                number: githubStats?.mergedPRs?.toString() || "3", 
                label: "PRs Merged", 
                testid: "stat-prs" 
              },
            ].map((stat, index) => (
              <motion.div
                key={index}
                whileHover={{ scale: 1.05, boxShadow: "0 0 25px rgba(59,130,246,0.3)" }}
                transition={{ duration: 0.3 }}
                className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 md:p-8 text-center"
                data-testid={stat.testid}
              >
                <div className="text-3xl sm:text-4xl md:text-5xl font-black text-blue-400 mb-2 leading-tight">
                  {stat.number}
                </div>
                <div className="text-sm md:text-base text-gray-400 font-medium">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Contributors Section */}
      <section className="py-20 px-4 bg-gradient-to-b from-black/50 to-black border-t border-white/10">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4" data-testid="text-contributors-heading">
            Built by the Community
          </h2>
          <p className="text-center text-gray-400 mb-12 max-w-2xl mx-auto">
            Meet the developers, engineers, and AI enthusiasts building the future of Linux
          </p>

          {contributorsLoading ? (
            // Loading skeleton
            <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-4">
              {Array.from({ length: 20 }).map((_, i) => (
                <div
                  key={i}
                  className="aspect-square rounded-xl bg-white/5 animate-pulse"
                  data-testid={`contributor-skeleton-${i}`}
                />
              ))}
            </div>
          ) : (
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-100px" }}
              variants={{
                visible: {
                  transition: {
                    staggerChildren: 0.03,
                  },
                },
              }}
              className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-4"
            >
              {contributors?.map((contributor, index) => (
                <motion.a
                  key={contributor.login}
                  href={contributor.html_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  variants={{
                    hidden: { opacity: 0, scale: 0.8 },
                    visible: { opacity: 1, scale: 1 },
                  }}
                  whileHover={{ scale: 1.1, zIndex: 10 }}
                  transition={{ duration: 0.2 }}
                  className="group relative aspect-square rounded-xl overflow-hidden border border-white/10 hover:border-blue-400 hover:shadow-[0_0_20px_rgba(59,130,246,0.4)] transition-all duration-300"
                  data-testid={`contributor-${contributor.login}`}
                >
                  <img
                    src={contributor.avatar_url}
                    alt={contributor.login}
                    loading="lazy"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-2">
                    <div className="text-xs font-semibold text-white truncate">
                      {contributor.login}
                    </div>
                    <div className="text-xs text-blue-400">
                      {contributor.contributions} commits
                    </div>
                  </div>
                </motion.a>
              ))}
            </motion.div>
          )}

          {/* Call to action */}
          <div className="text-center mt-12">
            <p className="text-gray-400 mb-4">Want to contribute?</p>
            <a
              href="https://github.com/cortexlinux/cortex"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 bg-white/5 backdrop-blur-xl border border-white/10 rounded-lg hover:border-blue-400 hover:shadow-[0_0_20px_rgba(59,130,246,0.3)] transition-all duration-300"
              data-testid="button-contribute"
            >
              <Github size={20} />
              <span>Start Contributing on GitHub</span>
              <ChevronRight size={16} />
            </a>
          </div>
        </div>
      </section>

      {/* Problem Section */}
      <section className="py-20 px-4 bg-black border-t border-white/10">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16" data-testid="text-tired-fighting">
            Tired of Fighting Your OS?
          </h2>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={{
              visible: {
                transition: {
                  staggerChildren: 0.1,
                },
              },
            }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12"
          >
            {[
              {
                text: "47 Stack Overflow tabs to install CUDA",
                testid: "pain-stackoverflow",
              },
              {
                text: "Days wasted on dependency conflicts",
                testid: "pain-dependencies",
              },
              {
                text: '"Works on my machine" syndrome',
                testid: "pain-works-on-my-machine",
              },
              {
                text: "Configuration files in ancient runes",
                testid: "pain-config-files",
              },
            ].map((pain, index) => (
              <motion.div
                key={index}
                variants={{
                  hidden: { opacity: 0, x: -20 },
                  visible: { opacity: 1, x: 0 },
                }}
                className="flex items-start gap-4 bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6"
                data-testid={pain.testid}
              >
                <div className="w-10 h-10 bg-red-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                  <X size={24} className="text-red-400" />
                </div>
                <p className="text-lg text-gray-300">{pain.text}</p>
              </motion.div>
            ))}
          </motion.div>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4 }}
            className="text-center text-xl text-gray-400 max-w-3xl mx-auto"
            data-testid="text-developers-waste"
          >
            Developers waste 30% of their time fighting the OS instead of building.
          </motion.p>
        </div>
      </section>

      {/* Solution Section */}
      <section className="py-20 px-4 bg-gradient-to-b from-black to-blue-950/10">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-6" data-testid="text-meet-cortex">
            Meet Cortex: Your AI System Administrator
          </h2>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={{
              visible: {
                transition: {
                  staggerChildren: 0.15,
                },
              },
            }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-16"
          >
            {[
              {
                icon: MessageSquare,
                title: "Natural Language Commands",
                description:
                  "Tell Cortex what you need in plain English. It understands intent, not just keywords.",
                example: "cortex install tensorflow --optimize-gpu",
                testid: "solution-natural-language",
              },
              {
                icon: Wrench,
                title: "Hardware-Aware Optimization",
                description:
                  "Automatically detects your GPU, CPU, and memory. Configures software for maximum performance.",
                example: "Detects NVIDIA RTX 4090 → Installs CUDA 12.3",
                testid: "solution-hardware-aware",
              },
              {
                icon: RefreshCcw,
                title: "Self-Healing Configuration",
                description:
                  "Fixes broken dependencies automatically. Rollback if anything goes wrong. Never repeat errors.",
                example: "Dependency conflict? Cortex resolves it.",
                testid: "solution-self-healing",
              },
            ].map((feature, index) => (
              <motion.div
                key={index}
                variants={{
                  hidden: { opacity: 0, y: 30 },
                  visible: { opacity: 1, y: 0 },
                }}
                className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 hover:border-blue-400 hover:shadow-[0_0_25px_rgba(59,130,246,0.3)] transition-all duration-300"
                data-testid={feature.testid}
              >
                <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6">
                  <feature.icon size={32} className="text-blue-400" />
                </div>
                <h3 className="text-2xl font-bold mb-4">{feature.title}</h3>
                <p className="text-gray-400 mb-4 leading-relaxed">{feature.description}</p>
                <div className="bg-black/50 border border-white/10 rounded-lg p-3">
                  <code className="text-sm text-green-400">{feature.example}</code>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Security Section */}
      <section id="security" className="py-20 px-4 bg-gradient-to-b from-blue-950/10 to-black border-t border-white/10">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16" data-testid="text-enterprise-security">
            Enterprise-Grade Security
          </h2>

          <div className="bg-gradient-to-br from-blue-950/30 to-purple-950/30 backdrop-blur-xl border border-blue-400/30 rounded-2xl p-8 md:p-12">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, margin: "-100px" }}
              variants={{
                visible: {
                  transition: {
                    staggerChildren: 0.1,
                  },
                },
              }}
              className="grid grid-cols-1 md:grid-cols-2 gap-8"
            >
              {[
                {
                  icon: Lock,
                  title: "Sandboxed Execution",
                  description:
                    "AI never has direct kernel access. Every command runs in isolated Firejail container.",
                  testid: "security-sandbox",
                },
                {
                  icon: Eye,
                  title: "Preview Before Execute",
                  description:
                    "Review all commands before they run. You approve every system change.",
                  testid: "security-preview",
                },
                {
                  icon: RotateCcw,
                  title: "Instant Rollback",
                  description:
                    "Undo any change in seconds. Full system snapshots before major operations.",
                  testid: "security-rollback",
                },
                {
                  icon: FileText,
                  title: "Complete Audit Logging",
                  description:
                    "Track every command, every change. Full transparency for compliance.",
                  testid: "security-audit",
                },
              ].map((feature, index) => (
                <motion.div
                  key={index}
                  variants={{
                    hidden: { opacity: 0, y: 20 },
                    visible: { opacity: 1, y: 0 },
                  }}
                  className="flex gap-4"
                  data-testid={feature.testid}
                >
                  <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <feature.icon size={24} className="text-blue-400" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
                    <p className="text-gray-400 leading-relaxed">{feature.description}</p>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* Comparison Table Section */}
      <section className="py-20 px-4 bg-black border-t border-white/10">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16" data-testid="text-how-different">
            How Is Cortex Different?
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left p-4 md:p-6 text-sm md:text-base font-semibold" data-testid="table-header-feature">
                    Feature
                  </th>
                  <th className="text-center p-4 md:p-6 text-sm md:text-base font-semibold" data-testid="table-header-warp">
                    Warp/Gemini CLI
                  </th>
                  <th className="text-center p-4 md:p-6 text-sm md:text-base font-semibold" data-testid="table-header-claude">
                    Claude Code
                  </th>
                  <th className="text-center p-4 md:p-6 text-sm md:text-base font-semibold text-blue-400" data-testid="table-header-cortex">
                    Cortex Linux
                  </th>
                </tr>
              </thead>
              <tbody>
                {[
                  {
                    feature: "AI-assisted commands",
                    warp: true,
                    claude: true,
                    cortex: true,
                    testid: "row-ai-commands",
                  },
                  {
                    feature: "Hardware detection",
                    warp: false,
                    claude: false,
                    cortex: true,
                    testid: "row-hardware",
                  },
                  {
                    feature: "Dependency resolution",
                    warp: false,
                    claude: false,
                    cortex: true,
                    testid: "row-dependencies",
                  },
                  {
                    feature: "GPU optimization",
                    warp: false,
                    claude: false,
                    cortex: true,
                    testid: "row-gpu",
                  },
                  {
                    feature: "System configuration",
                    warp: false,
                    claude: false,
                    cortex: true,
                    testid: "row-system-config",
                  },
                  {
                    feature: "OS-level integration",
                    warp: false,
                    claude: false,
                    cortex: true,
                    testid: "row-os-integration",
                  },
                  {
                    feature: "Preview commands",
                    warp: true,
                    claude: true,
                    cortex: true,
                    testid: "row-preview",
                  },
                  {
                    feature: "Rollback capability",
                    warp: false,
                    claude: false,
                    cortex: true,
                    testid: "row-rollback",
                  },
                ].map((row, index) => (
                  <tr
                    key={index}
                    className="border-b border-white/10 last:border-0"
                    data-testid={row.testid}
                  >
                    <td className="p-4 md:p-6 text-sm md:text-base text-gray-300">
                      {row.feature}
                    </td>
                    <td className="p-4 md:p-6 text-center">
                      {row.warp ? (
                        <Check size={20} className="text-green-400 mx-auto" />
                      ) : (
                        <X size={20} className="text-red-400 mx-auto" />
                      )}
                    </td>
                    <td className="p-4 md:p-6 text-center">
                      {row.claude ? (
                        <Check size={20} className="text-green-400 mx-auto" />
                      ) : (
                        <X size={20} className="text-red-400 mx-auto" />
                      )}
                    </td>
                    <td className="p-4 md:p-6 text-center">
                      {row.cortex ? (
                        <Check size={20} className="text-blue-400 mx-auto" />
                      ) : (
                        <X size={20} className="text-red-400 mx-auto" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Use Cases Section - Replaces "How It Works" */}
      <section className="py-20 px-4 bg-gradient-to-b from-black to-blue-950/10 border-t border-white/10">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16" data-testid="text-real-problems">
            Real Problems, Real Solutions
          </h2>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={{
              visible: {
                transition: {
                  staggerChildren: 0.15,
                },
              },
            }}
            className="grid grid-cols-1 md:grid-cols-2 gap-8"
          >
            {[
              {
                role: "Data Scientists",
                before: "6 hours installing CUDA + TensorFlow + dependencies across 47 Stack Overflow tabs",
                after: "cortex install tensorflow --optimize-gpu (5 minutes)",
                timeSaved: "5h 55m",
                testid: "usecase-data-scientists",
              },
              {
                role: "DevOps Engineers",
                before: "4 hours configuring Oracle DB with manual dependency resolution",
                after: "cortex setup oracle-23-ai production-ready (4 minutes)",
                timeSaved: "3h 56m",
                testid: "usecase-devops",
              },
              {
                role: "ML Engineers",
                before: "Version conflicts between PyTorch and CUDA, 3 hours debugging",
                after: "cortex install pytorch stable --compatible-cuda (automatic resolution)",
                timeSaved: "3h",
                testid: "usecase-ml-engineers",
              },
              {
                role: "Students",
                before: '"Works on my machine" but crashes on professor\'s system',
                after: "Reproducible environments, exact dependency versions",
                timeSaved: "Frustration: Eliminated",
                testid: "usecase-students",
              },
            ].map((useCase, index) => (
              <motion.div
                key={index}
                variants={{
                  hidden: { opacity: 0, y: 30 },
                  visible: { opacity: 1, y: 0 },
                }}
                className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 hover:border-blue-400 hover:shadow-[0_0_25px_rgba(59,130,246,0.3)] transition-all duration-300"
                data-testid={useCase.testid}
              >
                <h3 className="text-2xl font-bold mb-6 text-blue-400">FOR {useCase.role.toUpperCase()}</h3>

                <div className="space-y-4 mb-6">
                  <div className="flex gap-3">
                    <X size={24} className="text-red-400 flex-shrink-0 mt-1" />
                    <div>
                      <p className="text-sm font-semibold text-gray-400 mb-1">Before:</p>
                      <p className="text-gray-300">{useCase.before}</p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <Check size={24} className="text-green-400 flex-shrink-0 mt-1" />
                    <div>
                      <p className="text-sm font-semibold text-gray-400 mb-1">With Cortex:</p>
                      <p className="text-gray-300">{useCase.after}</p>
                    </div>
                  </div>
                </div>

                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                  <p className="text-green-400 font-semibold text-center">
                    Time Saved: {useCase.timeSaved}
                  </p>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-20 px-4 bg-black">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16" data-testid="text-what-makes-different">
            What Makes Cortex Different?
          </h2>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={{
              visible: {
                transition: {
                  staggerChildren: 0.1,
                },
              },
            }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
          >
            {[
              {
                icon: Brain,
                title: "Natural Language Control",
                description:
                  "Tell Cortex what you want in plain English. 'Install Docker for development' - and it handles dependencies, configuration, and security automatically.",
                testid: "card-natural-language",
              },
              {
                icon: Shield,
                title: "Enterprise-Grade Security",
                description:
                  "Every command runs in isolated environments with AppArmor and Firejail. AI-validated actions prevent mistakes before they happen.",
                testid: "card-security",
              },
              {
                icon: TrendingUp,
                title: "Learns Your Workflow",
                description:
                  "Cortex remembers your patterns, detects repetitive tasks, and suggests automation. It gets smarter with every interaction.",
                testid: "card-learning",
              },
              {
                icon: Package,
                title: "Smart Package Management",
                description:
                  "AI wrapper over apt that understands package relationships, resolves conflicts, and suggests optimal installation strategies.",
                testid: "card-package",
              },
              {
                icon: RotateCcw,
                title: "Time-Travel System Recovery",
                description:
                  "Complete installation history with one-click rollback. Made a mistake? Revert to any previous system state instantly.",
                testid: "card-recovery",
              },
              {
                icon: Github,
                title: "Open Source",
                description:
                  "Open source from day one. Built by developers, for developers. Contribute features and earn bounties.",
                testid: "card-open-source",
              },
            ].map((feature, index) => (
              <motion.div
                key={index}
                variants={{
                  hidden: { opacity: 0, y: 30 },
                  visible: { opacity: 1, y: 0 },
                }}
                className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 transition-all duration-300 hover:scale-[1.02] hover:-translate-y-2 hover:border-blue-400 hover:shadow-[0_0_20px_rgba(59,130,246,0.3)] group"
                data-testid={feature.testid}
              >
                <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6 transition-all duration-300 group-hover:shadow-[0_0_30px_rgba(59,130,246,0.6),0_0_50px_rgba(234,179,8,0.4)]">
                  <feature.icon size={32} className="text-blue-400" />
                </div>
                <h3 className="text-xl md:text-2xl font-bold mb-4">{feature.title}</h3>
                <p className="text-base text-gray-400 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Join Section */}
      <section id="join" className="py-20 px-4 bg-black">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16" data-testid="text-build-future">
            Build the Future of Linux
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
            {/* Left Column: Developer Benefits */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 h-full">
              <h3 className="text-2xl md:text-3xl font-bold mb-6" data-testid="text-for-developers">
                For Developers
              </h3>

              <div className="space-y-4 mb-8">
                {[
                  "Earn bounties for contributions ($25-$500 per feature)",
                  "Work on cutting-edge AI + systems programming",
                  "Join a global community of innovators",
                  "2x bonus payment when we close seed funding",
                  "Early contributor opportunities at a $50-100B market",
                ].map((benefit, index) => (
                  <div key={index} className="flex items-start gap-3" data-testid={`benefit-${index}`}>
                    <CheckCircle size={24} className="text-blue-400 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-300 text-base md:text-lg">{benefit}</span>
                  </div>
                ))}
              </div>

              <a
                href="https://github.com/cortexlinux/cortex/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="w-full px-8 py-4 bg-blue-500 rounded-lg font-semibold text-center hover:shadow-[0_0_25px_rgba(59,130,246,0.5)] hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2"
                data-testid="button-view-issues"
              >
                <Github size={20} />
                View Open Issues on GitHub
                <ExternalLink size={20} />
              </a>
            </div>

            {/* Right Column: Current Opportunities */}
            <div>
              <h3 className="text-2xl md:text-3xl font-bold mb-6" data-testid="text-current-opportunities">
                Current Opportunities
              </h3>

              <div className="grid grid-cols-1 gap-4">
                {(githubIssues && githubIssues.length > 0 ? githubIssues : [
                  {
                    title: "AI Context Memory System",
                    bounty: "$200",
                    skills: ["Python", "ML/AI", "SQLite"],
                    difficulty: "Advanced",
                    url: "https://github.com/cortexlinux/cortex/issues",
                  },
                  {
                    title: "Network & Proxy Config",
                    bounty: "$150",
                    skills: ["Python", "Networking"],
                    difficulty: "Medium",
                    url: "https://github.com/cortexlinux/cortex/issues",
                  },
                  {
                    title: "Plugin System",
                    bounty: "$200",
                    skills: ["Python", "Architecture"],
                    difficulty: "Advanced",
                    url: "https://github.com/cortexlinux/cortex/issues",
                  },
                  {
                    title: "User Preferences System",
                    bounty: "$100",
                    skills: ["Python", "Config"],
                    difficulty: "Beginner",
                    url: "https://github.com/cortexlinux/cortex/issues",
                  },
                ]).map((issue: any, index: number) => (
                  <div
                    key={index}
                    className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-xl p-6 hover:border-blue-400 hover:-translate-y-1 hover:shadow-[0_0_20px_rgba(59,130,246,0.3)] transition-all duration-300"
                    data-testid={`issue-${index}`}
                  >
                    <h4 className="text-lg md:text-xl font-bold mb-3">{issue.title}</h4>

                    <span className="inline-block px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-semibold mb-3">
                      {issue.bounty}
                    </span>

                    <div className="flex flex-wrap gap-2 mb-3">
                      {issue.skills.map((skill: string, skillIndex: number) => (
                        <span
                          key={skillIndex}
                          className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-xs"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>

                    <div className="mb-4">
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-semibold ${
                          issue.difficulty === "Advanced"
                            ? "bg-red-500/20 text-red-400"
                            : issue.difficulty === "Medium"
                            ? "bg-yellow-500/20 text-yellow-400"
                            : "bg-green-500/20 text-green-400"
                        }`}
                      >
                        {issue.difficulty}
                      </span>
                    </div>

                    <a
                      href={issue.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block w-full px-4 py-2 border-2 border-blue-400 rounded-lg text-sm font-semibold text-center hover:bg-blue-400/10 transition-all"
                      data-testid={`button-claim-${index}`}
                    >
                      View Issue on GitHub
                    </a>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Sponsor Section */}
      <section id="sponsor" className="py-20 px-4 bg-gradient-to-b from-black to-blue-950/10 border-t border-white/10">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16" data-testid="text-fund-innovation">
            Fund Open Source Innovation
          </h2>

          {/* Sponsor Tiers */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            {[
              {
                name: "Bronze",
                price: "$1,000",
                features: [
                  "Logo on README",
                  "Discord sponsor channel access",
                  "Monthly progress reports",
                ],
                buttonText: "Become Bronze Sponsor",
                featured: false,
                testid: "tier-bronze",
              },
              {
                name: "Silver",
                price: "$5,000",
                features: [
                  "Everything in Bronze",
                  "Priority feature requests",
                  "Quarterly strategy calls",
                  "Co-marketing opportunities",
                ],
                buttonText: "Become Silver Sponsor",
                featured: true,
                testid: "tier-silver",
              },
              {
                name: "Gold",
                price: "$10,000+",
                features: [
                  "Everything in Silver",
                  "Dedicated support channel",
                  "Early access to enterprise features",
                  "Joint case studies",
                  "Advisory board seat",
                ],
                buttonText: "Become Gold Sponsor",
                featured: false,
                testid: "tier-gold",
              },
            ].map((tier, index) => (
              <motion.div
                key={index}
                whileHover={{ scale: 1.02 }}
                className={`bg-white/5 backdrop-blur-xl ${
                  tier.featured ? "border-2 border-blue-400" : "border border-white/10"
                } rounded-2xl p-8 h-full relative hover:shadow-[0_0_25px_rgba(59,130,246,0.3)] transition-all duration-300`}
                data-testid={tier.testid}
              >
                {tier.featured && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 bg-blue-500 px-4 py-1 rounded-full text-xs font-bold uppercase">
                    Most Popular
                  </div>
                )}

                <h3 className="text-2xl md:text-3xl font-bold mb-2">{tier.name}</h3>
                <div className="text-4xl md:text-5xl font-black text-blue-400 mb-1">
                  {tier.price}
                </div>
                <p className="text-sm text-gray-400 mb-6">/month</p>

                <div className="space-y-3 mb-8">
                  {tier.features.map((feature, featureIndex) => (
                    <div key={featureIndex} className="flex items-start gap-2">
                      <CheckCircle size={20} className="text-blue-400 flex-shrink-0 mt-0.5" />
                      <span className="text-sm text-gray-300">{feature}</span>
                    </div>
                  ))}
                </div>

                <button
                  className={`w-full px-6 py-3 rounded-lg font-semibold transition-all ${
                    tier.featured
                      ? "bg-blue-500 hover:shadow-[0_0_25px_rgba(59,130,246,0.5)]"
                      : "border-2 border-blue-400 hover:bg-blue-400/10"
                  }`}
                  data-testid={`button-sponsor-${tier.name.toLowerCase()}`}
                >
                  {tier.buttonText}
                </button>
              </motion.div>
            ))}
          </div>

          {/* Enterprise Partnership */}
          <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border-2 border-blue-400/50 rounded-2xl p-8 md:p-12 text-center" data-testid="card-enterprise">
            <h3 className="text-3xl md:text-4xl font-bold mb-4">Enterprise Partnership</h3>
            <p className="text-lg text-gray-300 mb-6 max-w-3xl mx-auto">
              Building on Cortex Linux? Let's discuss custom partnership opportunities including:
              white-label licensing, dedicated development resources, SLA guarantees, and equity
              participation.
            </p>
            <button className="px-8 py-4 bg-blue-500 rounded-lg text-lg font-semibold hover:shadow-[0_0_25px_rgba(59,130,246,0.5)] hover:scale-105 transition-all duration-300 inline-flex items-center gap-2" data-testid="button-contact-enterprise">
              Contact for Enterprise
              <ArrowRight size={20} />
            </button>
          </div>
        </div>
      </section>

      {/* Business Model Section */}
      <section id="pricing" className="py-20 px-4 bg-gradient-to-b from-black to-blue-950/10 border-t border-white/10">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16" data-testid="text-choose-edition">
            Choose Your Edition
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Community Edition */}
            <motion.div
              whileHover={{ scale: 1.02 }}
              className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 hover:shadow-[0_0_25px_rgba(59,130,246,0.3)] transition-all duration-300"
              data-testid="edition-community"
            >
              <h3 className="text-3xl font-bold mb-2">Community Edition</h3>
              <div className="text-5xl font-black text-green-400 mb-4">FREE</div>
              <p className="text-gray-400 mb-6">Open Source Forever</p>

              <div className="space-y-3 mb-8">
                {[
                  "Full AI capabilities",
                  "All core features",
                  "Open source (Apache 2.0)",
                  "Community support",
                  "Unlimited personal use",
                ].map((feature, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <Check size={20} className="text-green-400 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-300">{feature}</span>
                  </div>
                ))}
              </div>

              <a
                href="https://github.com/cortexlinux/cortex"
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full px-6 py-3 border-2 border-green-400 rounded-lg font-semibold text-center hover:bg-green-400/10 transition-all"
                data-testid="button-download-community"
              >
                Download Free
              </a>
            </motion.div>

            {/* Enterprise Edition */}
            <motion.div
              whileHover={{ scale: 1.02 }}
              className="bg-white/5 backdrop-blur-xl border-2 border-blue-400 rounded-2xl p-8 hover:shadow-[0_0_25px_rgba(59,130,246,0.3)] transition-all duration-300 relative"
              data-testid="edition-enterprise"
            >
              <div className="absolute -top-4 left-1/2 -translate-x-1/2 bg-blue-500 px-4 py-1 rounded-full text-xs font-bold uppercase">
                For Teams
              </div>

              <h3 className="text-3xl font-bold mb-2">Enterprise Edition</h3>
              <div className="text-5xl font-black text-blue-400 mb-4">Custom</div>
              <p className="text-gray-400 mb-6">Contact for Pricing</p>

              <div className="space-y-3 mb-8">
                {[
                  "Everything in Community",
                  "Priority support (24/7)",
                  "Compliance reporting",
                  "Role-based access control",
                  "Custom integrations",
                  "SLA guarantees",
                ].map((feature, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <Check size={20} className="text-blue-400 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-300">{feature}</span>
                  </div>
                ))}
              </div>

              <button
                className="w-full px-6 py-3 bg-blue-500 rounded-lg font-semibold hover:shadow-[0_0_25px_rgba(59,130,246,0.5)] transition-all"
                data-testid="button-contact-enterprise-edition"
              >
                Contact Sales
              </button>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Community Section */}
      <section className="py-20 px-4 bg-black">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-5xl font-bold mb-12" data-testid="text-join-community">
            Join the Community
          </h2>

          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-12 hover:shadow-[0_0_25px_rgba(59,130,246,0.3)] transition-all duration-300"
            data-testid="card-discord"
          >
            <div className="w-20 h-20 mx-auto mb-6 text-blue-400 drop-shadow-[0_0_15px_rgba(59,130,246,0.6)]">
              <MessageCircle size={80} />
            </div>

            <h3 className="text-2xl md:text-3xl font-bold mb-3">Join Our Discord</h3>
            <p className="text-gray-400 mb-6">
              Real-time collaboration, bounty discussions, and technical support
            </p>

            <a
              href="https://discord.gg/uCqHvxjU83"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block px-8 py-4 bg-blue-500 rounded-lg text-lg font-semibold hover:shadow-[0_0_25px_rgba(59,130,246,0.5)] hover:scale-105 transition-all duration-300"
              data-testid="button-join-discord"
            >
              Join Discord
            </a>
          </motion.div>

          {/* Social Links */}
          <div className="flex flex-wrap justify-center gap-6 mt-12">
            {[
              {
                icon: Github,
                label: "github.com/cortexlinux/cortex",
                href: "https://github.com/cortexlinux/cortex",
                testid: "link-social-github",
              },
              {
                icon: Mail,
                label: "mike@cortexlinux.com",
                href: "mailto:mike@cortexlinux.com",
                testid: "link-social-email",
              },
              {
                icon: Globe,
                label: "cortexlinux.com",
                href: "https://cortexlinux.com",
                testid: "link-social-website",
              },
            ].map((social, index) => (
              <a
                key={index}
                href={social.href}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-gray-400 hover:text-blue-400 transition-colors"
                data-testid={social.testid}
              >
                <social.icon size={20} />
                <span className="text-sm">{social.label}</span>
              </a>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-20 px-4 bg-gradient-to-b from-black to-blue-950/10 border-t border-white/10">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-16" data-testid="text-beta-testers">
            What Beta Testers Are Saying
          </h2>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            variants={{
              visible: {
                transition: {
                  staggerChildren: 0.15,
                },
              },
            }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8"
          >
            {[
              {
                quote: "This saved me 20 hours on my last project setup.",
                author: "Beta Tester",
                testid: "testimonial-1",
              },
              {
                quote: "Finally, Linux that doesn't fight back.",
                author: "Beta Tester",
                testid: "testimonial-2",
              },
              {
                quote: "Security features give me confidence to use AI at system level.",
                author: "Beta Tester",
                testid: "testimonial-3",
              },
            ].map((testimonial, index) => (
              <motion.div
                key={index}
                variants={{
                  hidden: { opacity: 0, y: 30 },
                  visible: { opacity: 1, y: 0 },
                }}
                className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 hover:border-blue-400 hover:shadow-[0_0_25px_rgba(59,130,246,0.3)] transition-all duration-300"
                data-testid={testimonial.testid}
              >
                <div className="mb-6">
                  <User size={40} className="text-blue-400" />
                </div>
                <p className="text-lg text-gray-300 mb-6 italic">"{testimonial.quote}"</p>
                <p className="text-sm text-gray-400">— {testimonial.author}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-black border-t border-white/10 py-12 px-4">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8 text-center md:text-left">
          {/* Left: Branding */}
          <div>
            <div className="text-2xl font-bold mb-2">
              <span className="text-white">CORTEX</span>{" "}
              <span className="text-blue-400">LINUX</span>
            </div>
            <p className="text-sm text-gray-400 mb-1">AI-Native Operating System</p>
            <p className="text-xs text-gray-500">Built by AI Venture Holdings LLC</p>
          </div>

          {/* Middle: Resources */}
          <div>
            <h4 className="text-sm font-semibold mb-4">Resources</h4>
            <div className="space-y-2">
              {[
                { label: "Home", onClick: () => onNavigate("home"), testid: "footer-link-home" },
                { label: "FAQ", href: "/faq", testid: "footer-link-faq" },
                { label: "GitHub", href: "https://github.com/cortexlinux/cortex", testid: "footer-link-github" },
                { label: "Discord", href: "https://discord.gg/uCqHvxjU83", testid: "footer-link-discord" },
                { label: "Blog", href: "#", testid: "footer-link-blog" },
                { label: "Contact", href: "mailto:mike@cortexlinux.com", testid: "footer-link-contact" },
              ].map((link, index) => (
                <div key={index}>
                  {link.onClick ? (
                    <button
                      onClick={link.onClick}
                      className="text-sm text-gray-400 hover:text-blue-400 transition-colors text-left"
                      data-testid={link.testid}
                    >
                      {link.label}
                    </button>
                  ) : (
                    <a
                      href={link.href}
                      target={link.href?.startsWith("http") ? "_blank" : undefined}
                      rel={link.href?.startsWith("http") ? "noopener noreferrer" : undefined}
                      className="text-sm text-gray-400 hover:text-blue-400 transition-colors block"
                      data-testid={link.testid}
                    >
                      {link.label}
                    </a>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Right: Info */}
          <div className="space-y-2">
            <p className="text-sm text-gray-400">
              Contact: <a href="mailto:mike@cortexlinux.com" className="text-blue-400 hover:underline">mike@cortexlinux.com</a>
            </p>
            <p className="text-sm text-gray-400">Seeking $2-3M seed funding</p>
            <p className="text-sm text-gray-400">Launching February 2025</p>
            <p className="text-sm text-gray-400">© 2025 AI Venture Holdings LLC</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
