import { NextAuthOptions } from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'
import { compare } from 'bcrypt'

// Extended types for NextAuth
declare module 'next-auth' {
  interface User {
    role: string
    subscription: string
  }
  interface Session {
    user: {
      id: string
      name?: string | null
      email?: string | null
      image?: string | null
      role: string
      subscription: string
    }
  }
}

declare module 'next-auth/jwt' {
  interface JWT {
    role: string
    subscription: string
  }
}

// This is a simplified auth configuration for demonstration
// In production, this would integrate with your backend API

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: 'credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null
        }

        // Demo authentication - replace with real API call
        const demoUsers = [
          {
            id: '1',
            email: 'admin@sheily.ai',
            password: 'demo', // In production, this would be hashed
            name: 'System Administrator',
            role: 'admin',
            subscription: 'enterprise'
          }
        ]

        const user = demoUsers.find(u => u.email === credentials.email)

        if (user && credentials.password === user.password) {
          return {
            id: user.id,
            email: user.email,
            name: user.name,
            role: user.role,
            subscription: user.subscription,
          }
        }

        return null
      }
    })
  ],
  pages: {
    signIn: '/auth/signin',
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.role = user.role
        token.subscription = user.subscription
      }
      return token
    },
    async session({ session, token }) {
      if (token) {
        session.user.id = token.sub!
        session.user.role = token.role as string
        session.user.subscription = token.subscription as string
      }
      return session
    },
  },
  session: {
    strategy: 'jwt',
    maxAge: 24 * 60 * 60, // 24 hours
  },
  secret: process.env.NEXTAUTH_SECRET || 'your-secret-key',
}
