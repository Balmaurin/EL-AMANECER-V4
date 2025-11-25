import { NextAuthOptions } from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        // Por ahora, aceptar cualquier credencial para desarrollo
        // En producción, esto debería validar contra una base de datos
        if (credentials?.username && credentials?.password) {
          return {
            id: '1',
            name: credentials.username,
            email: `${credentials.username}@sheily.ai`,
          }
        }
        return null
      }
    })
  ],
  pages: {
    signIn: '/auth/signin',
  },
  session: {
    strategy: 'jwt',
  },
  secret: process.env.NEXTAUTH_SECRET || 'sheily-secret-key-change-in-production',
}
