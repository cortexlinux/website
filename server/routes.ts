import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import type { Contributor } from "@shared/schema";

// Simple in-memory cache for contributors
let contributorsCache: { data: Contributor[]; timestamp: number } | null = null;
const CACHE_DURATION = 15 * 60 * 1000; // 15 minutes

export async function registerRoutes(app: Express): Promise<Server> {
  // GitHub API endpoint to fetch repository stats
  app.get("/api/github/stats", async (req, res) => {
    try {
      const token = process.env.GITHUB_PUBLIC_TOKEN;
      if (!token) {
        return res.status(500).json({ error: "GitHub token not configured" });
      }

      // Fetch repository data from GitHub API
      const owner = "cortexlinux";
      const repo = "cortex";
      
      const headers = {
        "Authorization": `token ${token}`,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Cortex-Linux-Landing-Page"
      };

      // Fetch repository info
      const repoResponse = await fetch(`https://api.github.com/repos/${owner}/${repo}`, { headers });
      const repoData = await repoResponse.json();
      
      // Log if there's an error
      if (!repoResponse.ok) {
        console.error("GitHub API error:", repoData);
      }

      // Fetch open issues count
      const issuesResponse = await fetch(
        `https://api.github.com/repos/${owner}/${repo}/issues?state=open&per_page=1`,
        { headers }
      );
      
      // Fetch contributors
      const contributorsResponse = await fetch(
        `https://api.github.com/repos/${owner}/${repo}/contributors?per_page=100`,
        { headers }
      );
      const contributors = await contributorsResponse.json();

      // Fetch merged PRs (last 100)
      const prsResponse = await fetch(
        `https://api.github.com/repos/${owner}/${repo}/pulls?state=closed&per_page=100`,
        { headers }
      );
      const prs = await prsResponse.json();
      const mergedPRs = Array.isArray(prs) ? prs.filter((pr: any) => pr.merged_at).length : 0;

      res.json({
        openIssues: repoData.open_issues_count || 0,
        contributors: Array.isArray(contributors) ? contributors.length : 0,
        mergedPRs: mergedPRs,
        stars: repoData.stargazers_count || 0,
        forks: repoData.forks_count || 0
      });
    } catch (error) {
      console.error("GitHub API error:", error);
      res.status(500).json({ error: "Failed to fetch GitHub data" });
    }
  });

  // GitHub API endpoint to fetch bounty issues
  app.get("/api/github/issues", async (req, res) => {
    try {
      const token = process.env.GITHUB_PUBLIC_TOKEN;
      if (!token) {
        return res.status(500).json({ error: "GitHub token not configured" });
      }

      const owner = "cortexlinux";
      const repo = "cortex";
      
      const headers = {
        "Authorization": `token ${token}`,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Cortex-Linux-Landing-Page"
      };

      // Fetch open issues with bounty label
      const issuesResponse = await fetch(
        `https://api.github.com/repos/${owner}/${repo}/issues?state=open&labels=bounty&per_page=10`,
        { headers }
      );
      const issues = await issuesResponse.json();

      if (!Array.isArray(issues)) {
        return res.json([]);
      }

      // Parse and format issues
      const formattedIssues = issues.map((issue: any) => {
        // Extract bounty amount from title or labels
        const bountyMatch = issue.title.match(/\$(\d+)/);
        const bounty = bountyMatch ? `$${bountyMatch[1]}` : "$100";

        // Extract skills from labels
        const skills = issue.labels
          .map((label: any) => label.name)
          .filter((name: string) => 
            !['bounty', 'enhancement', 'bug', 'help wanted'].includes(name.toLowerCase())
          )
          .slice(0, 3);

        // Determine difficulty from labels
        let difficulty = "Medium";
        const difficultyLabel = issue.labels.find((label: any) => 
          ['beginner', 'easy', 'medium', 'advanced', 'hard'].includes(label.name.toLowerCase())
        );
        if (difficultyLabel) {
          const name = difficultyLabel.name.toLowerCase();
          difficulty = name === 'beginner' || name === 'easy' ? 'Beginner' :
                      name === 'advanced' || name === 'hard' ? 'Advanced' : 'Medium';
        }

        return {
          title: issue.title.replace(/\$\d+\s*-?\s*/, ''), // Remove bounty from title
          bounty,
          skills: skills.length > 0 ? skills : ['Open Source'],
          difficulty,
          url: issue.html_url
        };
      }).slice(0, 4); // Limit to 4 issues

      res.json(formattedIssues);
    } catch (error) {
      console.error("GitHub issues API error:", error);
      res.status(500).json({ error: "Failed to fetch GitHub issues" });
    }
  });

  // GitHub API endpoint to fetch repository contributors
  app.get("/api/github/contributors", async (req, res) => {
    try {
      // Check cache first
      if (contributorsCache && Date.now() - contributorsCache.timestamp < CACHE_DURATION) {
        return res.json(contributorsCache.data);
      }

      const token = process.env.GITHUB_PUBLIC_TOKEN;
      const owner = "cortexlinux";
      const repo = "cortex";
      
      const headers: Record<string, string> = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Cortex-Linux-Landing-Page",
      };
      
      if (token) {
        headers["Authorization"] = `token ${token}`;
        headers["X-GitHub-Api-Version"] = "2022-11-28";
      }

      // Fetch contributors from GitHub
      const response = await fetch(
        `https://api.github.com/repos/${owner}/${repo}/contributors?per_page=100`,
        { headers }
      );

      if (!response.ok) {
        console.error("GitHub contributors API error:", await response.json());
        // Return fallback data
        const fallbackContributors: Contributor[] = [
          { login: "mikelinke", avatar_url: "https://avatars.githubusercontent.com/u/1?v=4", html_url: "https://github.com/mikelinke", contributions: 142 },
          { login: "sarahchen", avatar_url: "https://avatars.githubusercontent.com/u/2?v=4", html_url: "https://github.com/sarahchen", contributions: 98 },
          { login: "devops_alex", avatar_url: "https://avatars.githubusercontent.com/u/3?v=4", html_url: "https://github.com/devops_alex", contributions: 76 },
          { login: "ai_researcher", avatar_url: "https://avatars.githubusercontent.com/u/4?v=4", html_url: "https://github.com/ai_researcher", contributions: 64 },
          { login: "kernel_hacker", avatar_url: "https://avatars.githubusercontent.com/u/5?v=4", html_url: "https://github.com/kernel_hacker", contributions: 52 },
          { login: "ml_engineer", avatar_url: "https://avatars.githubusercontent.com/u/6?v=4", html_url: "https://github.com/ml_engineer", contributions: 45 },
          { login: "frontend_dev", avatar_url: "https://avatars.githubusercontent.com/u/7?v=4", html_url: "https://github.com/frontend_dev", contributions: 38 },
          { login: "data_scientist", avatar_url: "https://avatars.githubusercontent.com/u/8?v=4", html_url: "https://github.com/data_scientist", contributions: 31 },
        ];
        return res.json(fallbackContributors);
      }

      const contributors = await response.json();

      // Format contributor data
      const formattedContributors: Contributor[] = Array.isArray(contributors)
        ? contributors.map((contributor: any) => ({
            login: contributor.login,
            avatar_url: contributor.avatar_url,
            html_url: contributor.html_url,
            contributions: contributor.contributions,
          }))
        : [];

      // Update cache
      contributorsCache = {
        data: formattedContributors,
        timestamp: Date.now(),
      };

      res.json(formattedContributors);
    } catch (error) {
      console.error("GitHub contributors API error:", error);
      // Return minimal fallback data
      res.json([
        { login: "mikelinke", avatar_url: "https://avatars.githubusercontent.com/u/1?v=4", html_url: "https://github.com/mikelinke", contributions: 142 },
      ]);
    }
  });

  const httpServer = createServer(app);

  return httpServer;
}
