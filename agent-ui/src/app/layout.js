import './globals.css'

export const metadata = {
  title: 'Hacksy - AI Agents Recommender',
  description: 'AI-powered hackathon and agent recommendations based on your GitHub profile using advanced AI',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  )
}
