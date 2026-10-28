import React from 'react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import { cn } from '@/utils/cn';

export const Drawer = DialogPrimitive.Root;
export const DrawerTrigger = DialogPrimitive.Trigger;
export const DrawerPortal = DialogPrimitive.Portal;
export const DrawerClose = DialogPrimitive.Close;

export function DrawerOverlay({
  className,
  ...props
}: React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>) {
  return (
    <DialogPrimitive.Overlay
      className={cn(
        'fixed inset-0 z-50 bg-ink/30 backdrop-blur-[1px] animate-in fade-in duration-150',
        className
      )}
      {...props}
    />
  );
}

export function DrawerContent({
  className,
  children,
  ...props
}: React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>) {
  return (
    <DrawerPortal>
      <DrawerOverlay />
      <DialogPrimitive.Content
        className={cn(
          'fixed right-0 top-0 bottom-0 z-50 w-full max-w-md bg-surface border-l border-line p-6 shadow-popover flex flex-col duration-150 animate-in slide-in-from-right',
          className
        )}
        {...props}
      >
        <div className="flex items-center justify-between pb-4 border-b border-line mb-4">
          <div id="drawer-header" />
          <DialogPrimitive.Close className="rounded p-1 text-muted hover:text-ink hover:bg-paper focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forest">
            <X className="w-4 h-4" />
            <span className="sr-only">Close drawer</span>
          </DialogPrimitive.Close>
        </div>
        <div className="flex-1 overflow-y-auto">{children}</div>
      </DialogPrimitive.Content>
    </DrawerPortal>
  );
}
