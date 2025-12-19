import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Link } from "wouter";
import {
  Send,
  Copy,
  Check,
  Terminal,
  Sparkles,
  Github,
  ChevronLeft,
  Loader2,
  AlertCircle,
  Zap,
} from "lucide-react";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { useCortexDemo } from "@/hooks/useCortexDemo";

const EXAMPLE_PROMPTS = [
  "install docker with compose",
  "set up python for machine learning",
  "configure nginx as reverse proxy",
  "install nodejs and npm",
  "set up postgresql database",
];

export default function BetaPage() {
  const { toast } = useToast();
  const [input, setInput] = useState("");
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const terminalRef = useRef<HTMLDivElement>(null);

  const { messages, sendMessage, isLoading, error, remaining, limitReached, clearMessages } = useCortexDemo();

  // Scroll terminal to bottom on new messages
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || limitReached) return;

    const userInput = input;
    setInput("");
    await sendMessage(userInput);
  };

  const handleExampleClick = async (prompt: string) => {
    if (isLoading || limitReached) return;
    setInput("");
    await sendMessage(prompt);
  };

  const copyToClipboard = async (text: string, idx: number) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIdx(idx);
      toast({ title: "Copied to clipboard" });
      setTimeout(() => setCopiedIdx(null), 2000);
    } catch {
      toast({ title: "Failed to copy", variant: "destructive" });
    }
  };

  const extractCommands = (text: string): string => {
    // Extract code blocks or return raw text
    const codeBlockMatch = text.match(/```(?:bash|sh)?\n?([\s\S]*?)```/);
    if (codeBlockMatch) return codeBlockMatch[1].trim();
    return text;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white">
                <ChevronLeft className="w-4 h-4 mr-1" />
                Back
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <Terminal className="w-6 h-6 text-orange-500" />
              <span className="text-xl font-bold text-white">Cortex</span>
              <span className="px-2 py-0.5 text-xs bg-orange-500/20 text-orange-400 rounded-full border border-orange-500/30">
                BETA
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {remaining !== null && (
              <span className="text-sm text-gray-400">
                {remaining} requests remaining
              </span>
            )}
            <a href="https://github.com/cortexlinux/cortex" target="_blank" rel="noopener noreferrer">
              <Button variant="outline" size="sm" className="border-gray-700 text-gray-300 hover:bg-gray-800">
                <Github className="w-4 h-4 mr-2" />
                GitHub
              </Button>
            </a>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Try Cortex <span className="text-orange-500">Live</span>
          </h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Tell Cortex what you want to install or configure in plain English.
            Get the exact commands you need.
          </p>
        </motion.div>

        {/* Limit Reached Banner */}
        {limitReached && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-6"
          >
            <Card className="bg-orange-500/10 border-orange-500/30">
              <CardContent className="py-6 text-center">
                <Zap className="w-12 h-12 text-orange-500 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-white mb-2">Demo Limit Reached</h3>
                <p className="text-gray-300 mb-4">
                  You've used all 5 demo requests. Install Cortex to get unlimited access!
                </p>
                <Link href="/install">
                  <Button className="bg-orange-500 hover:bg-orange-600 text-white">
                    Install Cortex
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Terminal */}
        <Card className="bg-gray-900/50 border-gray-800 mb-6">
          <CardHeader className="border-b border-gray-800 py-3">
            <div className="flex items-center gap-2">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <div className="w-3 h-3 rounded-full bg-yellow-500" />
                <div className="w-3 h-3 rounded-full bg-green-500" />
              </div>
              <span className="text-sm text-gray-400 ml-2">cortex-demo</span>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div
              ref={terminalRef}
              className="h-80 overflow-y-auto p-4 font-mono text-sm space-y-4"
            >
              {messages.length === 0 ? (
                <div className="text-gray-500 text-center py-8">
                  <Sparkles className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>Try an example below or type your own request</p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} className="space-y-2">
                    {msg.role === "user" ? (
                      <div className="flex items-start gap-2">
                        <span className="text-green-400">$</span>
                        <span className="text-gray-300">{msg.content}</span>
                      </div>
                    ) : (
                      <div className="bg-gray-800/50 rounded-lg p-3 relative group">
                        <pre className="text-orange-400 whitespace-pre-wrap text-xs md:text-sm overflow-x-auto">
                          {extractCommands(msg.content)}
                        </pre>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={() => copyToClipboard(extractCommands(msg.content), idx)}
                        >
                          {copiedIdx === idx ? (
                            <Check className="w-4 h-4 text-green-400" />
                          ) : (
                            <Copy className="w-4 h-4 text-gray-400" />
                          )}
                        </Button>
                      </div>
                    )}
                  </div>
                ))
              )}
              {isLoading && (
                <div className="flex items-center gap-2 text-gray-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Generating commands...</span>
                </div>
              )}
              {error && (
                <div className="flex items-center gap-2 text-red-400">
                  <AlertCircle className="w-4 h-4" />
                  <span>{error}</span>
                </div>
              )}
            </div>

            {/* Input */}
            <form onSubmit={handleSubmit} className="border-t border-gray-800 p-4">
              <div className="flex gap-2">
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="What do you want to install or configure?"
                  className="bg-gray-800 border-gray-700 text-white placeholder:text-gray-500"
                  disabled={isLoading || limitReached}
                />
                <Button
                  type="submit"
                  disabled={!input.trim() || isLoading || limitReached}
                  className="bg-orange-500 hover:bg-orange-600 text-white"
                >
                  {isLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Example Prompts */}
        <div className="mb-8">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Try these examples:</h3>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_PROMPTS.map((prompt) => (
              <Button
                key={prompt}
                variant="outline"
                size="sm"
                className="border-gray-700 text-gray-300 hover:bg-gray-800 hover:text-white"
                onClick={() => handleExampleClick(prompt)}
                disabled={isLoading || limitReached}
              >
                {prompt}
              </Button>
            ))}
          </div>
        </div>

        {/* Info Cards */}
        <div className="grid md:grid-cols-2 gap-4 mb-8">
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white text-lg flex items-center gap-2">
                <Terminal className="w-5 h-5 text-orange-500" />
                What is Cortex?
              </CardTitle>
            </CardHeader>
            <CardContent className="text-gray-400 text-sm">
              Cortex is an AI-powered package manager for Debian/Ubuntu Linux.
              Instead of memorizing apt commands, just describe what you need in plain English.
            </CardContent>
          </Card>
          <Card className="bg-gray-900/50 border-gray-800">
            <CardHeader>
              <CardTitle className="text-white text-lg flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-orange-500" />
                Full Installation
              </CardTitle>
            </CardHeader>
            <CardContent className="text-gray-400 text-sm">
              This demo shows command generation. The full Cortex CLI automatically
              executes commands with safety checks, rollback support, and more.
              <Link href="/install">
                <Button variant="link" className="text-orange-500 p-0 h-auto mt-2">
                  Install Cortex →
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {messages.length > 0 && (
          <div className="text-center">
            <Button
              variant="ghost"
              size="sm"
              onClick={clearMessages}
              className="text-gray-500 hover:text-gray-300"
            >
              Clear conversation
            </Button>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}
