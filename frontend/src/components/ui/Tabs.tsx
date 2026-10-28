import React from 'react';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import { cn } from '@/utils/cn';

export const Tabs = TabsPrimitive.Root;

export function TabsList({
  className,
  ...props
}: React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      className={cn(
        'inline-flex items-center gap-1 border-b border-line w-full bg-transparent p-0 text-muted',
        className
      )}
      {...props}
    />
  );
}

export function TabsTrigger({
  className,
  children,
  ...props
}: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      className={cn(
        'inline-flex items-center justify-center whitespace-nowrap px-4 py-2.5 text-table font-medium transition-colors duration-150 relative -mb-[1px]',
        'text-muted hover:text-ink hover:border-b-2 hover:border-line focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest',
        'data-[state=active]:text-forest data-[state=active]:font-semibold data-[state=active]:border-b-2 data-[state=active]:border-forest',
        className
      )}
      {...props}
    >
      {children}
    </TabsPrimitive.Trigger>
  );
}

export function TabsContent({
  className,
  ...props
}: React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>) {
  return (
    <TabsPrimitive.Content
      className={cn(
        'mt-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest',
        className
      )}
      {...props}
    />
  );
}
