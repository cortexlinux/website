import { motion } from "framer-motion";
import { Link } from "wouter";
import { ArrowRight, Calendar, Clock, Search, ChevronLeft } from "lucide-react";
import { useState, useMemo } from "react";
import { blogPosts, BlogPost } from "@/data/blogPosts";
import { Input } from "@/components/ui/input";

export default function Blog() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const categories = useMemo(() => {
    const cats = new Set(blogPosts.map(post => post.category));
    return Array.from(cats);
  }, []);

  const filteredPosts = useMemo(() => {
    return blogPosts
      .filter(post => {
        const matchesSearch = 
          post.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
          post.excerpt.toLowerCase().includes(searchTerm.toLowerCase()) ||
          post.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
        const matchesCategory = !selectedCategory || post.category === selectedCategory;
        return matchesSearch && matchesCategory;
      })
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  }, [searchTerm, selectedCategory]);

  return (
    <div className="min-h-screen pt-20 pb-16">
      {/* Header */}
      <div className="max-w-6xl mx-auto px-4 mb-12">
        <Link href="/" className="inline-flex items-center gap-2 text-gray-400 hover:text-blue-400 transition-colors mb-8">
          <ChevronLeft size={16} />
          Back to Home
        </Link>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="text-white">Cortex</span>{" "}
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">Blog</span>
          </h1>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Technical insights, tutorials, and best practices for ML engineers and developers
          </p>
        </motion.div>
      </div>

      {/* Search and Filters */}
      <div className="max-w-6xl mx-auto px-4 mb-8">
        <div className="flex flex-col md:flex-row gap-4 items-center">
          <div className="relative flex-1 w-full md:max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
            <Input
              type="text"
              placeholder="Search articles..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 bg-white/5 border-white/10 focus:border-blue-500/50"
              data-testid="input-blog-search"
            />
          </div>
          
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedCategory(null)}
              className={`px-3 py-1.5 rounded-full text-sm transition-all ${
                !selectedCategory 
                  ? "bg-blue-500/20 border border-blue-500/50 text-blue-300" 
                  : "bg-white/5 border border-white/10 text-gray-400 hover:border-white/20"
              }`}
              data-testid="filter-all"
            >
              All
            </button>
            {categories.map(category => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-3 py-1.5 rounded-full text-sm transition-all ${
                  selectedCategory === category 
                    ? "bg-blue-500/20 border border-blue-500/50 text-blue-300" 
                    : "bg-white/5 border border-white/10 text-gray-400 hover:border-white/20"
                }`}
                data-testid={`filter-${category.toLowerCase().replace(' ', '-')}`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Blog Posts Grid */}
      <div className="max-w-6xl mx-auto px-4">
        {filteredPosts.length === 0 ? (
          <div className="text-center py-16">
            <p className="text-gray-400">No posts found matching your search.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredPosts.map((post, index) => (
              <motion.div
                key={post.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <BlogCard post={post} />
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function BlogCard({ post }: { post: BlogPost }) {
  return (
    <Link href={`/blog/${post.slug}`}>
      <article
        className="group h-full rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden transition-all duration-300 hover:border-blue-500/30 hover:shadow-[0_0_30px_rgba(59,130,246,0.15)] hover:-translate-y-1 cursor-pointer"
        data-testid={`blog-card-${post.slug}`}
      >
        {post.image && (
          <div className="relative h-48 overflow-hidden">
            <img
              src={post.image}
              alt={post.title}
              className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
              loading="lazy"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
            <span className="absolute bottom-3 left-3 px-2 py-1 text-xs font-medium bg-blue-500/20 border border-blue-500/30 rounded-full text-blue-300">
              {post.category}
            </span>
          </div>
        )}
        
        <div className="p-5">
          {!post.image && (
            <span className="inline-block mb-3 px-2 py-1 text-xs font-medium bg-blue-500/10 border border-blue-500/20 rounded-full text-blue-400">
              {post.category}
            </span>
          )}
          
          <h3 className="text-lg font-semibold mb-2 text-white group-hover:text-blue-300 transition-colors line-clamp-2">
            {post.title}
          </h3>
          
          <p className="text-sm text-gray-400 mb-4 line-clamp-3">
            {post.excerpt}
          </p>
          
          <div className="flex items-center justify-between text-xs text-gray-500">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <Calendar size={12} />
                {new Date(post.date).toLocaleDateString('en-US', { 
                  month: 'short', 
                  day: 'numeric',
                  year: 'numeric'
                })}
              </span>
              <span className="flex items-center gap-1">
                <Clock size={12} />
                {post.readingTime}
              </span>
            </div>
            <span className="flex items-center gap-1 text-blue-400 group-hover:gap-2 transition-all">
              Read
              <ArrowRight size={12} />
            </span>
          </div>
        </div>
      </article>
    </Link>
  );
}
