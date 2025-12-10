import { useParams, Link, useLocation } from "wouter";
import { motion } from "framer-motion";
import { ChevronLeft, Calendar, Clock, User, Tag, ArrowRight } from "lucide-react";
import { getPostBySlug, getRelatedPosts, BlogPost } from "@/data/blogPosts";
import { useEffect } from "react";

export default function BlogPostPage() {
  const { slug } = useParams<{ slug: string }>();
  const [, setLocation] = useLocation();
  const post = slug ? getPostBySlug(slug) : undefined;
  const relatedPosts = slug ? getRelatedPosts(slug, 2) : [];

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [slug]);

  if (!post) {
    return (
      <div className="min-h-screen pt-20 pb-16 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-white mb-4">Post not found</h1>
          <Link href="/blog" className="text-blue-400 hover:underline">
            Back to Blog
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-20 pb-16">
      <article className="max-w-4xl mx-auto px-4">
        {/* Back link */}
        <Link href="/blog" className="inline-flex items-center gap-2 text-gray-400 hover:text-blue-400 transition-colors mb-8">
          <ChevronLeft size={16} />
          Back to Blog
        </Link>

        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <span className="inline-block mb-4 px-3 py-1 text-sm font-medium bg-blue-500/10 border border-blue-500/20 rounded-full text-blue-400">
            {post.category}
          </span>
          
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-white mb-6 leading-tight">
            {post.title}
          </h1>
          
          <div className="flex flex-wrap items-center gap-4 text-sm text-gray-400">
            <span className="flex items-center gap-2">
              <User size={14} />
              {post.author}
            </span>
            <span className="flex items-center gap-2">
              <Calendar size={14} />
              {new Date(post.date).toLocaleDateString('en-US', { 
                month: 'long', 
                day: 'numeric',
                year: 'numeric'
              })}
            </span>
            <span className="flex items-center gap-2">
              <Clock size={14} />
              {post.readingTime}
            </span>
          </div>
        </motion.header>

        {/* Featured Image */}
        {post.image && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-10 rounded-xl overflow-hidden"
          >
            <img
              src={post.image}
              alt={post.title}
              className="w-full h-64 md:h-80 object-cover"
            />
          </motion.div>
        )}

        {/* Content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="prose prose-invert prose-lg max-w-none mb-12"
        >
          <div 
            className="blog-content"
            dangerouslySetInnerHTML={{ __html: formatContent(post.content) }}
          />
        </motion.div>

        {/* Tags */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="flex flex-wrap items-center gap-2 mb-12 pb-12 border-b border-white/10"
        >
          <Tag size={14} className="text-gray-500" />
          {post.tags.map(tag => (
            <span
              key={tag}
              className="px-3 py-1 text-sm bg-white/5 border border-white/10 rounded-full text-gray-400"
            >
              {tag}
            </span>
          ))}
        </motion.div>

        {/* Related Posts */}
        {relatedPosts.length > 0 && (
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <h2 className="text-2xl font-bold text-white mb-6">Related Articles</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {relatedPosts.map(related => (
                <RelatedPostCard key={related.id} post={related} />
              ))}
            </div>
          </motion.section>
        )}
      </article>
    </div>
  );
}

function RelatedPostCard({ post }: { post: BlogPost }) {
  return (
    <Link href={`/blog/${post.slug}`}>
      <article className="group p-5 rounded-xl border border-white/10 bg-white/[0.02] transition-all duration-300 hover:border-blue-500/30 hover:shadow-[0_0_20px_rgba(59,130,246,0.1)] cursor-pointer">
        <span className="inline-block mb-2 px-2 py-0.5 text-xs font-medium bg-blue-500/10 rounded text-blue-400">
          {post.category}
        </span>
        <h3 className="font-semibold text-white group-hover:text-blue-300 transition-colors mb-2 line-clamp-2">
          {post.title}
        </h3>
        <p className="text-sm text-gray-400 line-clamp-2 mb-3">
          {post.excerpt}
        </p>
        <span className="inline-flex items-center gap-1 text-sm text-blue-400 group-hover:gap-2 transition-all">
          Read more
          <ArrowRight size={14} />
        </span>
      </article>
    </Link>
  );
}

function formatContent(content: string): string {
  return content
    .replace(/^## (.+)$/gm, '<h2 class="text-2xl font-bold text-white mt-10 mb-4">$1</h2>')
    .replace(/^### (.+)$/gm, '<h3 class="text-xl font-semibold text-white mt-8 mb-3">$1</h3>')
    .replace(/^#### (.+)$/gm, '<h4 class="text-lg font-semibold text-white mt-6 mb-2">$1</h4>')
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
    .replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 bg-white/10 rounded text-blue-300 text-sm font-mono">$1</code>')
    .replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => 
      `<pre class="bg-black/50 border border-white/10 rounded-lg p-4 overflow-x-auto my-6"><code class="text-sm font-mono text-gray-300">${code.trim()}</code></pre>`
    )
    .replace(/^- (.+)$/gm, '<li class="text-gray-300 ml-4 list-disc">$1</li>')
    .replace(/^(\d+)\. (.+)$/gm, '<li class="text-gray-300 ml-4 list-decimal">$2</li>')
    .replace(/^- \[ \] (.+)$/gm, '<li class="text-gray-400 ml-4 flex items-center gap-2"><span class="w-4 h-4 border border-gray-500 rounded"></span>$1</li>')
    .replace(/^- \[x\] (.+)$/gm, '<li class="text-gray-300 ml-4 flex items-center gap-2"><span class="w-4 h-4 bg-blue-500/20 border border-blue-500 rounded flex items-center justify-center text-blue-400 text-xs">&#10003;</span>$1</li>')
    .replace(/\n\n/g, '</p><p class="text-gray-300 leading-relaxed mb-4">')
    .replace(/^(.+)$/gm, (match) => {
      if (match.startsWith('<')) return match;
      return `<p class="text-gray-300 leading-relaxed mb-4">${match}</p>`;
    });
}
