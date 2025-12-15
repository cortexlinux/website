import { motion } from "framer-motion";
import { Link } from "wouter";
import { ChevronLeft, Shield, Bug, Award, CheckCircle, AlertCircle, Mail } from "lucide-react";

export default function SecurityPage() {
  const sections = [
    {
      icon: Shield,
      title: "Security Practices",
      content: `Cortex Linux is built with security as a core principle:

Infrastructure Security:
• All data encrypted at rest (AES-256) and in transit (TLS 1.3)
• Regular security audits and penetration testing
• Automated vulnerability scanning in CI/CD pipeline
• Principle of least privilege for all system components

Application Security:
• Sandboxed command execution using Firejail containers
• Input validation and sanitization for all user inputs
• Secure secret management with encrypted storage
• Regular dependency updates and security patches

Code Security:
• Mandatory code review for all changes
• Static analysis and security linting
• Signed commits and releases
• Reproducible builds for verification`
    },
    {
      icon: Bug,
      title: "Vulnerability Disclosure",
      content: `We appreciate responsible disclosure of security vulnerabilities.

How to Report:
1. Email security@cortexlinux.com with details of the vulnerability
2. Include steps to reproduce, potential impact, and any proof-of-concept
3. Allow up to 90 days for us to address the issue before public disclosure

What to Expect:
• Acknowledgment within 48 hours of your report
• Regular updates on our progress
• Credit in our security advisories (if desired)
• No legal action for good-faith security research

What Not to Do:
• Access, modify, or delete data belonging to others
• Disrupt service availability
• Social engineering of employees or users
• Physical security testing`
    },
    {
      icon: Award,
      title: "Bug Bounty Program",
      content: `We reward security researchers who help keep Cortex Linux secure.

Scope:
• Cortex Linux core software and CLI
• Official web properties (cortexlinux.com)
• API endpoints and authentication systems

Rewards:
• Critical vulnerabilities: Up to $5,000
• High severity: Up to $2,500
• Medium severity: Up to $1,000
• Low severity: Up to $250

Eligibility:
• First reporter of a previously unknown vulnerability
• Vulnerability must be in-scope and valid
• Must follow responsible disclosure guidelines
• Must not be a current or recent employee

Submit reports to: security@cortexlinux.com`
    },
    {
      icon: CheckCircle,
      title: "Compliance",
      content: `Cortex Linux maintains compliance with industry standards:

SOC 2 Type II:
• Annual audits by independent third parties
• Controls for security, availability, and confidentiality
• Continuous monitoring and reporting

GDPR Compliance:
• Data processing agreements with all vendors
• Privacy by design principles
• User rights management (access, deletion, portability)
• Data Protection Impact Assessments

Additional Standards:
• ISO 27001 aligned security controls
• OWASP secure development practices
• CIS benchmark hardening guidelines

Enterprise customers can request compliance documentation and audit reports.`
    },
    {
      icon: AlertCircle,
      title: "Incident Response",
      content: `Our incident response process ensures rapid and effective handling of security events:

Detection:
• 24/7 automated monitoring and alerting
• Anomaly detection for unusual activity
• Regular log analysis and threat hunting

Response:
• Immediate containment of identified threats
• Root cause analysis and remediation
• Communication to affected users within 72 hours
• Post-incident review and improvements

Communication:
• Security advisories published on our website
• Email notifications for affected users
• Transparent incident reports
• GitHub security advisories for code vulnerabilities

Our target response times:
• Critical: 4 hours
• High: 24 hours
• Medium: 72 hours
• Low: 7 days`
    },
    {
      icon: Mail,
      title: "Security Contact",
      content: `For security-related inquiries:

Email: security@cortexlinux.com
PGP Key: Available at cortexlinux.com/security.asc

For general inquiries, please use:
• GitHub Issues: github.com/cortexlinux/cortex/issues
• Discord: discord.gg/cortexlinux

Enterprise Security:
Enterprise customers with specific security requirements can contact us for:
• Custom security assessments
• Dedicated security contacts
• Enhanced SLA for security issues
• Compliance documentation and attestations`
    }
  ];

  return (
    <div className="min-h-screen pt-20 pb-16">
      <div className="max-w-4xl mx-auto px-4">
        <Link href="/" className="inline-flex items-center gap-2 text-gray-400 hover:text-blue-400 transition-colors mb-8" data-testid="link-back-home">
          <ChevronLeft size={16} />
          Back to Home
        </Link>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="text-white">Security</span>{" "}
            <span className="text-gray-400">Policy</span>
          </h1>
          <p className="text-gray-400 text-lg">
            How we protect Cortex Linux and your systems
          </p>
        </motion.div>

        <div className="space-y-8">
          {sections.map((section, index) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white/5 border border-white/10 rounded-xl p-6"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-blue-500/10 rounded-lg">
                  <section.icon size={20} className="text-blue-400" />
                </div>
                <h2 className="text-xl font-semibold text-white">{section.title}</h2>
              </div>
              <div className="text-gray-400 whitespace-pre-line leading-relaxed text-sm">
                {section.content}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
