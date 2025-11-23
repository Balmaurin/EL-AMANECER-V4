'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// 2025 Modern Icons and Components
const GlassCard = ({ children, className = '', delay = 0 }: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 20, scale: 0.95 }}
    animate={{ opacity: 1, y: 0, scale: 1 }}
    transition={{ duration: 0.6, delay }}
    className={`bg-white/10 backdrop-blur-xl border border-white/20 rounded-3xl p-6 shadow-2xl ${className}`}
  >
    {children}
  </motion.div>
);

const MetricCard = ({ title, value, icon, trend, color, delay = 0 }: {
  title: string;
  value: string;
  icon: string;
  trend?: string;
  color?: string;
  delay?: number;
}) => (
  <GlassCard className="relative overflow-hidden group cursor-pointer">
    <div className={`absolute inset-0 bg-gradient-to-r ${color || 'from-blue-500/20 to-purple-500/20'} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />

    <div className="relative z-10">
      <div className="flex items-center justify-between mb-4">
        <div className={`text-3xl p-3 rounded-2xl bg-white/20`}>{icon}</div>
        {trend && (
          <div className={`text-sm px-3 py-1 rounded-full ${
            trend.startsWith('+') ? 'bg-emerald-500/20 text-emerald-300' : 'bg-red-500/20 text-red-300'
          }`}>
            {trend}
          </div>
        )}
      </div>

      <div className="space-y-2">
        <h3 className="text-white/80 text-sm font-medium">{title}</h3>
        <div className="text-3xl font-bold text-white">{value}</div>
      </div>
    </div>
  </GlassCard>
);

const NavigationTabs = ({ tabs, activeTab, onTabChange }: {
  tabs: string[];
  activeTab: string;
  onTabChange: (tab: string) => void;
}) => (
  <div className="flex space-x-1 mb-8 p-1 bg-white/10 rounded-2xl backdrop-blur-sm">
    {tabs.map((tab) => (
      <button
        key={tab}
        onClick={() => onTabChange(tab)}
        className={`relative px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
          activeTab === tab
            ? 'text-white'
            : 'text-white/70 hover:text-white'
        }`}
      >
        {activeTab === tab && (
          <motion.div
            layoutId="activeTab"
            className="absolute inset-0 bg-white/20 rounded-xl"
            transition={{ type: "spring", duration: 0.5 }}
          />
        )}
        <span className="relative z-10">{tab}</span>
      </button>
    ))}
  </div>
);

const FloatingParticles = () => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none">
    {[...Array(20)].map((_, i) => (
      <motion.div
        key={i}
        className="absolute w-1 h-1 bg-white/30 rounded-full"
        animate={{
          y: [0, -100, 0],
          opacity: [0, 1, 0],
        }}
        transition={{
          duration: Math.random() * 5 + 5,
          repeat: Infinity,
          delay: Math.random() * 5,
        }}
        style={{
          left: `${Math.random() * 100}%`,
          top: `${Math.random() * 100}%`,
        }}
      />
    ))}
  </div>
);

export default function ModernDashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [metrics, setMetrics] = useState({
    users: '2,847',
    agents: '42/47',
    conversations: '12,345',
    uptime: '99.8%'
  });

  const tabs = ['Overview', 'Analytics', 'Agents', 'RAG Studio'];

  const metricCards = [
    {
      title: 'Active Users',
      value: metrics.users,
      icon: '👥',
      trend: '+12.5%',
      color: 'from-cyan-500/20 to-teal-500/20'
    },
    {
      title: 'Agent Status',
      value: metrics.agents,
      icon: '🤖',
      trend: 'Active',
      color: 'from-emerald-500/20 to-green-500/20'
    },
    {
      title: 'Conversations',
      value: metrics.conversations,
      icon: '💬',
      trend: '+8.2%',
      color: 'from-violet-500/20 to-purple-500/20'
    },
    {
      title: 'System Uptime',
      value: metrics.uptime,
      icon: '⚡',
      trend: '99.8%',
      color: 'from-amber-500/20 to-orange-500/20'
    }
  ];

  return (
    <div className="min-h-screen relative">
      <FloatingParticles />

      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="text-center py-20 px-6 relative z-10"
      >
        <h1 className="text-6xl md:text-8xl font-bold mb-6">
          <span className="bg-gradient-to-r from-white via-cyan-200 to-purple-200 bg-clip-text text-transparent">
            Sheily AI
          </span>
        </h1>
        <p className="text-xl text-white/80 max-w-3xl mx-auto leading-relaxed">
          Enterprise-grade multi-agent orchestration platform powered by advanced AI,
          real-time analytics, and intelligent automation systems.
        </p>
      </motion.div>

      {/* Navigation */}
      <div className="flex justify-center mb-12">
        <NavigationTabs
          tabs={tabs}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      </div>

      {/* Dashboard Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'overview' && (
          <motion.div
            key="overview"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
            className="container mx-auto px-6 pb-20"
          >
            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
              {metricCards.map((card, index) => (
                <MetricCard
                  key={card.title}
                  {...card}
                  delay={index * 0.1}
                />
              ))}
            </div>

            {/* System Health Section */}
            <GlassCard className="mb-12">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">System Health</h2>
                <div className="flex space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400 text-sm font-medium">All Systems Operational</span>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-white/80">Performance Score</span>
                    <span className="text-green-400 font-bold">94%</span>
                  </div>
                  <div className="w-full bg-white/20 rounded-full h-2">
                    <motion.div
                      className="bg-gradient-to-r from-green-400 to-blue-500 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: "94%" }}
                      transition={{ duration: 1, delay: 0.5 }}
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-white/80">Security Score</span>
                    <span className="text-blue-400 font-bold">97%</span>
                  </div>
                  <div className="w-full bg-white/20 rounded-full h-2">
                    <motion.div
                      className="bg-gradient-to-r from-blue-400 to-purple-500 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: "97%" }}
                      transition={{ duration: 1, delay: 0.7 }}
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-white/80">Stability Score</span>
                    <span className="text-purple-400 font-bold">92%</span>
                  </div>
                  <div className="w-full bg-white/20 rounded-full h-2">
                    <motion.div
                      className="bg-gradient-to-r from-purple-400 to-pink-500 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: "92%" }}
                      transition={{ duration: 1, delay: 0.9 }}
                    />
                  </div>
                </div>
              </div>
            </GlassCard>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <GlassCard>
                <div className="text-center">
                  <div className="text-4xl mb-4">🚀</div>
                  <h3 className="text-xl font-bold text-white mb-2">Deploy Agent</h3>
                  <p className="text-white/70 mb-4">Launch a new AI agent with custom configurations</p>
                  <button className="bg-white/20 hover:bg-white/30 text-white px-6 py-2 rounded-xl transition-colors">
                    Get Started
                  </button>
                </div>
              </GlassCard>

              <GlassCard>
                <div className="text-center">
                  <div className="text-4xl mb-4">📊</div>
                  <h3 className="text-xl font-bold text-white mb-2">Analytics</h3>
                  <p className="text-white/70 mb-4">Deep dive into performance metrics and insights</p>
                  <button className="bg-white/20 hover:bg-white/30 text-white px-6 py-2 rounded-xl transition-colors">
                    View Reports
                  </button>
                </div>
              </GlassCard>

              <GlassCard>
                <div className="text-center">
                  <div className="text-4xl mb-4">⚙️</div>
                  <h3 className="text-xl font-bold text-white mb-2">Settings</h3>
                  <p className="text-white/70 mb-4">Configure system preferences and integrations</p>
                  <button className="bg-white/20 hover:bg-white/30 text-white px-6 py-2 rounded-xl transition-colors">
                    Configure
                  </button>
                </div>
              </GlassCard>
            </div>
          </motion.div>
        )}

        {activeTab === 'analytics' && (
          <motion.div
            key="analytics"
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.5 }}
            className="container mx-auto px-6 pb-20"
          >
            <GlassCard>
              <h2 className="text-3xl font-bold text-white mb-8">Advanced Analytics</h2>
              <div className="text-center py-20">
                <div className="text-6xl mb-6">📊</div>
                <h3 className="text-2xl font-bold text-white mb-4">Coming Soon</h3>
                <p className="text-white/70">Advanced analytics with 3D visualizations and real-time insights</p>
              </div>
            </GlassCard>
          </motion.div>
        )}

        {activeTab === 'agents' && (
          <motion.div
            key="agents"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -50 }}
            transition={{ duration: 0.5 }}
            className="container mx-auto px-6 pb-20"
          >
            <GlassCard>
              <h2 className="text-3xl font-bold text-white mb-8">AI Agents</h2>
              <div className="text-center py-20">
                <div className="text-6xl mb-6">🤖</div>
                <h3 className="text-2xl font-bold text-white mb-4">Agent Management</h3>
                <p className="text-white/70">Intelligent agent orchestration and deployment platform</p>
              </div>
            </GlassCard>
          </motion.div>
        )}

        {activeTab === 'rag' && (
          <motion.div
            key="rag"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.5 }}
            className="container mx-auto px-6 pb-20"
          >
            <GlassCard>
              <h2 className="text-3xl font-bold text-white mb-8">RAG Studio</h2>
              <div className="text-center py-20">
                <div className="text-6xl mb-6">🧠</div>
                <h3 className="text-2xl font-bold text-white mb-4">Retrieval Augmented Generation</h3>
                <p className="text-white/70">Advanced knowledge management and intelligent query processing</p>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
