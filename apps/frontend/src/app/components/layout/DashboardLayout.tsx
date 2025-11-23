'use client'

import { useState } from 'react'
import { useSession, signOut } from 'next-auth/react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  HomeIcon,
  CogIcon,
  UserGroupIcon,
  ChatBubbleLeftIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon,
  Bars3Icon,
  XMarkIcon,
  CubeIcon,
  CommandLineIcon
} from '@heroicons/react/24/outline'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon, current: true },
  { name: 'Agents', href: '/dashboard/agents', icon: CubeIcon, current: false },
  { name: 'Conversations', href: '/dashboard/conversations', icon: ChatBubbleLeftIcon, current: false },
  { name: 'Analytics', href: '/dashboard/analytics', icon: ChartBarIcon, current: false },
  { name: 'Users', href: '/dashboard/users', icon: UserGroupIcon, current: false },
  { name: 'API', href: '/dashboard/api', icon: CommandLineIcon, current: false },
  { name: 'Settings', href: '/dashboard/settings', icon: Cog6ToothIcon, current: false },
]

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { data: session } = useSession()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const pathname = usePathname()

  const handleSignOut = () => {
    signOut({ callbackUrl: '/auth/signin' })
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div className="fixed inset-0 bg-black bg-opacity-25" onClick={() => setSidebarOpen(false)} />

          {/* Sidebar */}
          <div className="relative flex w-full max-w-xs flex-1 flex-col bg-white pt-5 pb-4 shadow-xl">
            <div className="absolute top-0 right-0 -mr-12 pt-2">
              <button
                type="button"
                className="ml-1 flex h-10 w-10 items-center justify-center rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
                onClick={() => setSidebarOpen(false)}
              >
                <XMarkIcon className="h-6 w-6 text-white" />
              </button>
            </div>

            <div className="flex flex-shrink-0 items-center px-4">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center">
                    <CubeIcon className="h-5 w-5 text-white" />
                  </div>
                </div>
                <div className="ml-3">
                  <h1 className="text-lg font-semibold text-gray-900">Sheily MCP</h1>
                  <p className="text-sm text-gray-500">Enterprise Dashboard</p>
                </div>
              </div>
            </div>

            <div className="mt-5 h-0 flex-1 overflow-y-auto">
              <nav className="px-2 space-y-1">
                {navigation.map((item) => {
                  const isActive = pathname === item.href
                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                        isActive
                          ? 'bg-gray-100 text-gray-900'
                          : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                      }`}
                      onClick={() => setSidebarOpen(false)}
                    >
                      <item.icon
                        className={`mr-3 flex-shrink-0 h-5 w-5 ${
                          isActive ? 'text-gray-500' : 'text-gray-400 group-hover:text-gray-500'
                        }`}
                      />
                      {item.name}
                    </Link>
                  )
                })}
              </nav>
            </div>

            {/* Mobile user menu */}
            <div className="flex-shrink-0 flex border-t border-gray-200 p-4">
              <button
                onClick={handleSignOut}
                className="flex-shrink-0 group block w-full"
              >
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="h-10 w-10 rounded-full bg-gray-300 flex items-center justify-center">
                      {session?.user?.email?.[0].toUpperCase() || 'U'}
                    </div>
                  </div>
                  <div className="ml-3">
                    <p className="text-base font-medium text-gray-700 group-hover:text-gray-900 truncate">
                      {session?.user?.name || session?.user?.email || 'User'}
                    </p>
                    <div className="flex items-center text-sm text-gray-500">
                      <ArrowRightOnRectangleIcon className="mr-1 h-4 w-4" />
                      Sign out
                    </div>
                  </div>
                </div>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col">
        <div className="flex flex-col flex-grow bg-white pt-5 pb-4 overflow-y-auto border-r border-gray-200">
          <div className="flex items-center flex-shrink-0 px-4">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center">
                  <CubeIcon className="h-5 w-5 text-white" />
                </div>
              </div>
              <div className="ml-3">
                <h1 className="text-lg font-semibold text-gray-900">Sheily MCP</h1>
                <p className="text-sm text-gray-500">Enterprise Dashboard</p>
              </div>
            </div>
          </div>

          <div className="mt-8 flex-grow flex flex-col">
            <nav className="flex-1 px-2 pb-4 space-y-1">
              {navigation.map((item) => {
                const isActive = pathname === item.href
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`group flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                      isActive
                        ? 'bg-gray-100 text-gray-900'
                        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                    }`}
                  >
                    <item.icon
                      className={`mr-3 flex-shrink-0 h-5 w-5 ${
                        isActive ? 'text-gray-500' : 'text-gray-400 group-hover:text-gray-500'
                      }`}
                    />
                    {item.name}
                  </Link>
                )
              })}
            </nav>
          </div>

          {/* Desktop user menu */}
          <div className="flex-shrink-0 flex border-t border-gray-200 p-4">
            <button
              onClick={handleSignOut}
              className="flex-shrink-0 group block w-full"
            >
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="h-10 w-10 rounded-full bg-gray-300 flex items-center justify-center">
                    {session?.user?.email?.[0].toUpperCase() || 'U'}
                  </div>
                </div>
                <div className="ml-3">
                  <p className="text-base font-medium text-gray-700 group-hover:text-gray-900 truncate">
                    {session?.user?.name || session?.user?.email || 'User'}
                  </p>
                  <div className="flex items-center text-sm text-gray-500">
                    <ArrowRightOnRectangleIcon className="mr-1 h-4 w-4" />
                    Sign out
                  </div>
                </div>
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64 flex flex-col flex-1">
        {/* Top bar */}
        <div className="sticky top-0 z-10 flex-shrink-0 flex h-16 bg-white shadow border-b border-gray-200">
          <button
            type="button"
            className="px-4 border-r border-gray-200 text-gray-500 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500 lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Bars3Icon className="h-6 w-6" />
          </button>
          <div className="flex-1 px-4 flex justify-between">
            <div className="flex-1 flex">
              <div className="w-full flex md:ml-0">
                <div className="relative w-full text-gray-400 focus-within:text-gray-600">
                  {/* Placeholder for search - can be expanded */}
                </div>
              </div>
            </div>
            <div className="ml-4 flex items-center md:ml-6">
              {/* Profile dropdown placeholder */}
              <div className="relative ml-3">
                {/* Can add notifications or profile dropdown here */}
              </div>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1 relative overflow-y-auto focus:outline-none">
          <div className="py-6">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              {children}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
