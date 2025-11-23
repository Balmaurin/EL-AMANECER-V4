import { withAuth } from 'next-auth/middleware'
import { NextResponse } from 'next/server'

export default withAuth(
  function middleware(req) {
    // Add any additional middleware logic here
    const token = req.nextauth.token

    // Check if user has required role/permission
    if (!token) {
      return NextResponse.redirect(new URL('/auth/signin', req.url))
    }

    return NextResponse.next()
  },
  {
    callbacks: {
      authorized: ({ token, req }) => {
        // Protect these routes
        const protectedPaths = ['/dashboard', '/api/admin']

        const isProtectedPath = protectedPaths.some(path =>
          req.nextUrl.pathname.startsWith(path)
        )

        return !isProtectedPath || !!token
      },
    },
  }
)

export const config = {
  matcher: [
    '/dashboard/:path*',
    '/api/admin/:path*',
    // Exclude auth routes and static assets
    '/((?!auth/signin|auth/signup|_next/static|_next/image|favicon.ico|public).*)',
  ],
}
